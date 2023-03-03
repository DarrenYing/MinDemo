# MinDemo

### 文件说明
> 复现以下问题，ZeRO节约显存效果不及预期
> 
> 涉及代码文件为 
> 
> convlstm_2stage.py (dp2+pp2)，对应模型文件为 models/TwoStageConvLSTM.py
> 
> 训练过程的optimizer定义在 convlstm_2stage.py 的81行，ZeRO配置定义在 TwoStageConvLSTM.py 的186行
> 
> convlstm_dp.py (dp4)，对应模型文件为 models/BaseConvLSTM.py
> 
> 训练过程的optimizer定义在 convlstm_dp.py 的79行，ZeRO配置定义在 BaseConvLSTM.py 的87行

### 启动命令
```shell
# dp2 + pp2
python3 -m oneflow.distributed.launch --nproc_per_node 4 convlstm_2stage.py
# pp4
python3 -m oneflow.distributed.launch --nproc_per_node 4 convlstm_dp.py
```