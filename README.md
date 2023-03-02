# MinDemo

### 文件说明
> 复现以下问题，测试convlstm_2stage.py的流水线并行+数据并行
> 
> eager模式能正常运行，graph模式下，会报错，
> 报错信息如下

```
(GRAPH:ConvLSTMGraph_0:ConvLSTMGraph) start building plan.
F20230302 11:10:01.653455 49675 exec_graph.cpp:121]  Stack Grad expects the shape of input tensor is equal to like tensor's. , but got (8,19,8,50,50) at input and (8,16,25,50)at like
Error message from /home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/core/graph/exec_graph.cpp:121
        op_->InferBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx, &GlobalJobDesc()):  infer blob descs if failed, op name

  File "/home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/core/graph/exec_graph.cpp", line 121, in InferBlobDescs
    op_->InferBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx, &GlobalJobDesc())
  File "/home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/core/operator/operator.cpp", line 349, in InferBlobDescsIf
    InferOutBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx)
  File "/home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/core/operator/operator.cpp", line 357, in InferOutBlobDescsIf
    InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx)
  File "/home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/core/operator/user_op.cpp", line 671, in InferOutBlobDescs
    val_->physical_tensor_desc_infer_fn(&infer_ctx)
  File "/home/ci-user/runners/release/_work/oneflow/oneflow/oneflow/user/ops/stack_op.cpp", line 177, in InferLogicalTensorDesc

```
### 启动命令
```shell
python3 -m oneflow.distributed.launch --nproc_per_node 4 convlstm_2stage.py
```