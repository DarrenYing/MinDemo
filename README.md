# MinDemo

### 文件说明
> 复现以下bug，测试predrnn_base.py的数据并行
> 
> eager模式能正常运行，graph模式下，不使用数据并行，能正常运行;
> 
> 86行，```batch_data = flow.tensor(batch_data, dtype=flow.float32, placement=P01, sbp=S0)```
> 若将sbp改为flow.sbp.broadcast，也不会报错
> 
> 但```sbp=S0```，运行时在```start building plan```的```(48/60) CheckOpGraph```
> 过程中会报以下错误
> 

```
(GRAPH:PredRNNGraph_0:PredRNNGraph) start building plan.
F20230111 04:34:38.079638  4381 exec_graph.cpp:121]  Stack Grad expects the shape of input tensor is equal to like tensor's. , but got (4,19,8,100,100) at input and (4,16,50,100)at like
Error message from /home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/core/graph/exec_graph.cpp:121
        op_->InferBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx, &GlobalJobDesc()):  infer blob descs if failed, op name

  File "/home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/core/graph/exec_graph.cpp", line 121, in InferBlobDescs
    op_->InferBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx, &GlobalJobDesc())
  File "/home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/core/operator/operator.cpp", line 349, in InferBlobDescsIf
    InferOutBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx)
  File "/home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/core/operator/operator.cpp", line 357, in InferOutBlobDescsIf
    InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx)
  File "/home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/core/operator/user_op.cpp", line 786, in InferOutBlobDescs
    val_->physical_tensor_desc_infer_fn(&infer_ctx)
  File "/home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/user/ops/stack_op.cpp", line 178, in InferLogicalTensorDesc
```
### 启动命令
```shell
# graph模式运行
python3 -m oneflow.distributed.launch --nproc_per_node=2 predrnn_base.py
# eager模式运行
python3 -m oneflow.distributed.launch --nproc_per_node=2 predrnn_base.py --run_mode eager
```