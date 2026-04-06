⏺ AIBrix 里 PD（prefill/decode 分离）现在怎么扩容                                                                          
                                                                                                                           
  • AIBrix 的 PD“池”是用 StormService 的 roles 来表达的：一个 StormService 里可以同时有 prefill/decode 两个                
  role，各自有独立的 replicas（即两个 pool）。StormService 会把所有 RoleSet 的同名 role 聚合到                             
  status.roleStatuses（pkg/controller/stormservice/utils.go:313）。                                                        
  • 扩容/缩容由 PodAutoscaler（自定义                                                                                      
  CRD）驱动，支持三种策略：HPA/KPA/APA（api/autoscaling/v1alpha1/podautoscaler_types.go:81）。在 PD 场景里，关键是它支持   
  role                                                                                                                     
  级别缩放：spec.subTargetSelector.roleName=prefill|decode（api/autoscaling/v1alpha1/podautoscaler_types.go:60），控制器会 
  直接 patch 对应 role 的 replicas（pool mode：pkg/controller/podautoscaler/workload_scale.go:234）。                      
  • 网关侧的 PD 路由会把 ready pods 按 label 分成 prefill/decode，并尽量在同一个 roleset                                   
  内配对；同时用一些更接近“背压”的信号做负载不均衡处理（pkg/plugins/gateway/algorithms/pd_disaggregation.go:160），例如    
  decode 侧会读：                                                                                                          
    • RealtimeNumRequestsRunning（并发/排队 proxy）（pkg/plugins/gateway/algorithms/pd_disaggregation.go:328）             
    • AvgGenerationThroughputToksPerS（token 吞吐）（pkg/plugins/gateway/algorithms/pd_disaggregation.go:341）             
    • GPUCacheUsagePerc（cache 压力）（pkg/plugins/gateway/algorithms/pd_disaggregation.go:353）                           
                                                                                                                           
                                                                                                                           
  AIBrix 现有模式“能不能支持”两篇论文的做法                                                                                
                                                                                                                           
  • TokenScale（Token Velocity + Convertible Decoders）                                                                    
    • Token Velocity（领先指标）：AIBrix 已经有 token 吞吐相关指标（例如                                                   
  AvgGenerationThroughputToksPerS，pkg/metrics/metrics.go:618），也能把它们喂给 autoscaler；但 autoscaler                  
  侧的“趋势/速度”能力目前是空实现（pkg/controller/podautoscaler/metrics/client.go:264），所以没有开箱即用的 token-velocity 
  预测式扩容。                                                                                                             
    • Convertible Decoders（decode GPU 临时做 prefill）：AIBrix 当前的控制面是“不同 role 不同 PodTemplate + 调整           
  replicas”，并不支持“把一个 decode pod 原地变成 prefill pod”的角色转换状态机；要做 convertible decoder                    
  属于架构/运行时能力改造，不是改 autoscaler 参数就能实现。                                                                
  • HeteroScale（异构 + 联动扩容 + 拓扑/网络）                                                                             
    • AIBrix 有异构 GPU 的一些基础设施（例如通过外部服务/GPU optimizer 提供推荐值，MetricSourceType=external；外部 REST    
  endpoint 是可用的，pkg/controller/podautoscaler/metrics/fetcher.go:237），以及调度侧的策略（RoleSet 的 SchedulingStrategy
  走 godel/volcano/coscheduling）。                                                                                        
    • 但“用单一鲁棒指标对 prefill+decode 进行联动扩缩、维持架构平衡”这件事，AIBrix 目前更多是“两个 role 各配一个           
  PodAutoscaler，各自缩放”，缺少论文那种 coordinated policy 的一体化实现。                                                 
                                                                                                                           
                                                                                                                           
  把论文方法搬进 AIBrix，是否有必要/能否拿收益？                                                                           
                                                                                                                           
  • 可能有收益的典型场景：                                                                                                 
    • 线上流量非常 bursty，prefill 队列/背压出现得比 GPU 利用率更早，导致 TTFT/TPOT 经常被打爆——这时做 token-              
  velocity/背压领先指标，往往比“GPU util/请求数”更有效。                                                                   
    • prefill pod 冷启动/拉模型/初始化很慢——这时 TokenScale 的“Convertible                                                 
  Decoders”这类“快速缓冲”机制收益会更大（因为它绕开了启动延迟）。                                                          
  • 可能收益不大的场景：                                                                                                   
    • 负载比较平稳、你本来就预留了 headroom；或者瓶颈主要是上游限流/网络带宽/下游存储，而不是 prefill/decode 的算力失衡。  
  • 更务实的落地建议（通常 ROI 更高的顺序）：                                                                              
    • 先用 AIBrix 已有指标把扩容从“滞后”改成“更接近背压”：例如对 prefill 用等待/并发/TTFT 相关指标，对 decode 用           
  tokens/s、running requests、TPOT 相关指标（token 吞吐指标定义见 pkg/metrics/metrics.go:607）。                           
    • 如果仍然跟不上 burst，再考虑补齐 autoscaler 的                                                                       
  trend/velocity（pkg/controller/podautoscaler/metrics/client.go:264）做“预测式提前扩容”；这类改动对现有架构侵入相对小。   
    • Convertible decoder 属于最大工程量/最大不确定性的方案：只有在“启动慢 + burst 强 +                                    
  违约成本高”同时成立时，才更值得上强改造。                                                                                

 AIBrix 里 PD 分离（prefill/decode）现在怎么扩容？                                            
                                                                                               
  • PD 分离在 AIBrix 里通常用 StormService 的 roles 来表达（prefill/decode），例如             
  samples/disaggregation/vllm/pool.yaml:1、samples/disaggregation/vllm/1p1d.yaml:1。           
  • 扩缩容入口是自研 CRD PodAutoscaler：通过 spec.scaleTargetRef                               
  指向要扩的对象（StormService/RayClusterFleet/Deployment 等），并且可以用                     
  spec.subTargetSelector.roleName 精确到 StormService 的某个 role（prefill 或                  
  decode）单独扩容，见                                                                         
  api/autoscaling/v1alpha1/podautoscaler_types.go:53、pkg/controller/podautoscaler/workload_sca
  le.go:92。                                                                                   
  • 控制环路是 controller 每轮（默认 10s）拉取目标 pods 的指标、做窗口聚合、算出期望副本数并   
  patch 到目标资源上：关键流程在                                                               
  pkg/controller/podautoscaler/podautoscaler_controller.go:720、pkg/controller/podautoscaler/po
  dautoscaler_controller.go:971；StormService role 的副本数实际写回到                          
  StormService.spec.template.spec.roles[i].replicas（pool 模式）或                             
  StormService.spec.replicas（replica 模式），见                                               
  pkg/controller/podautoscaler/workload_scale.go:185。                                         
  • 指标来源支持 pod/resource/custom/external（也就是可以从每个 pod 的 /metrics、K8s metrics   
  API、自定义 metrics API、或外部服务拿指标），见                                              
  api/autoscaling/v1alpha1/podautoscaler_types.go:109；算法策略支持 HPA/KPA/APA（KPA 有        
  stable/panic window），见                                                                    
  api/autoscaling/v1alpha1/podautoscaler_types.go:96、pkg/controller/podautoscaler/algorithm/kp
  a.go:41。                                                                                    
  • PD 场景的“prefill、decode 各扩各的”是官方示例做法：两个 PodAutoscaler 分别绑到同一个       
  StormService 的不同 roleName，各用自己的指标（比如                                           
  prefill_queue_length、decode_batch_utilization），见 samples/autoscaling/stormservice-       
  pool.yaml:8；文档也明确支持这种 role-level autoscaling，见                                   
  docs/source/features/autoscaling/metric-based-autoscaling.rst:65。                           
                                                                                               
  AIBrix 的扩容模式能支持 TokenScale/HeteroScale 那些思路吗？                                  
                                                                                               
  • Token Velocity（领先指标）：AIBrix “能接入”，但“默认不自带”。原因是 PodAutoscaler          
  的指标是可插拔的（pod/external/custom                                                        
  都行，api/autoscaling/v1alpha1/podautoscaler_types.go:109），所以你可以把 Token Velocity     
  做成某个可观测指标（由 engine/gateway/外部聚合服务计算后暴露），然后直接让 PodAutoscaler     
  用它来扩容；但 AIBrix 代码里“趋势/速度”相关能力目前是 stub（还没实现），见                   
  pkg/controller/podautoscaler/metrics/client.go:264。                                         
  • prefill/decode 联动扩容（一个指标同时驱动两边、保持平衡）：AIBrix                          
  目前的原语更偏“每个目标一个 autoscaler”，PD 常见是 prefill 一个 PA、decode 一个 PA（如示例   
  samples/autoscaling/stormservice-pool.yaml:8）。它也支持单个 PA 多指标取                     
  max（docs/source/features/autoscaling/metric-based-autoscaling.rst:90，实现上是逐个 metric   
  算、取最大推荐值，pkg/controller/podautoscaler/autoscaler.go:126），但不等价于 HeteroScale   
  那种“用一个鲁棒指标协调两池并约束比例/平衡”的策略；那部分需要你在控制面额外加“协调逻辑”（新  
  controller 或在现有 controller 上扩展）。                                                    
  • Convertible Decoders（decoder 在突发时临时干 prefill）：AIBrix 的 autoscaler               
  本身不提供这种“角色可转换”的系统机制；不过你这里的 PD 部署里 prefill/decode 容器跑的是同一套 
  vLLM server（例如                                                                            
  samples/disaggregation/vllm/pool.yaml:1），理论上可以通过“调度/路由层”把部分 decode          
  实例临时也接 prefill 流量（相当于软转换），但这属于路由/EPP 侧能力与策略，不是现有           
  PodAutoscaler 直接能实现的。                                                                 
                                                                                               
  把两篇论文的扩容方法搬到 AIBrix，有必要吗？能拿到收益吗？                                    
                                                                                               
  • 如果你现在的痛点是“突发流量下 TTFT/TPOT 大幅抖动，prefill/decoder                          
  经常打架且扩容滞后”：值得做，收益大概率来自两点：1) 用更“领先/更贴近 backpressure”的指标替代 
  GPU 利用率/粗粒度请求数；2) 做 PD 两池的协调与快速缓冲（类似 convertible decoder             
  思路）。AIBrix 当前虽然支持用队列类指标（示例就是 prefill_queue_length），但没有内置 Token   
  Velocity 这种跨阶段统一指标与预测控制环。                                                    
  • 如果你已经在用队列长度、inflight tokens、batch utilization 这类“贴近排队压力”的指标分别扩  
  prefill/decode，且 SLO 已经比较稳：把 TokenScale/HeteroScale                                 
  完整搬进来可能是“边际收益有限、工程量不小”。这时更现实的收益往往来自调参（KPA/APA            
  的窗口、容忍区间、冷却时间等，pkg/controller/podautoscaler/types/annotations.go:21）和指标口 
  径统一，而不是大改控制理论。                                                                 
  • 落地建议（低成本到高成本）：                                                               
    • 低成本：优先把 PD 扩容指标从“资源利用率”切到“排队/背压类指标”，并让 prefill/decode       
  各用自己的指标独立扩（按 samples/autoscaling/stormservice-pool.yaml:8 的方式）。             
    • 中成本：把“Token Velocity”做成一个可观测指标（外部聚合服务或 runtime 导出），用          
  metricSourceType: external/pod 接到 AIBrix                                                   
  PodAutoscaler（api/autoscaling/v1alpha1/podautoscaler_types.go:109），先验证它相对 queue     
  length 是否更早、更稳定地反映背压。                                                          
    • 高成本：做“联动扩容 + 可转换角色”的闭环（需要路由/EPP 侧配合），否则只改 autoscaler      
  很难复现论文里“吸收 burst、避免 prefill 初始化延迟”的核心收益。                              
