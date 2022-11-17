# MinDemo

### 文件说明
> 单机模型为 BaseConvLSTM_V2.py，入口程序为 base_demo_V2.py
> 
> 2阶段流水线并行模型为 TwoStageConvLSTM_V2.py，入口程序为 demo_V2_2stage.py

### 启动命令
```shell
python3 -m oneflow.distributed.launch --nproc_per_node=1 base_demo_V2.py

python3 -m oneflow.distributed.launch --nproc_per_node=2 demo_V2_2stage.py
```