# MinDemo

### 文件说明
> 复现以下bug
> 
> eager模式能正常运行，使用graph后会报以下错误
> 
> F20221208 16:08:27.755779 19446 job_builder.cpp:336] Check failed: modified_op_conf_op_names_.emplace(op_conf.name()).second model.motion_list.0-add_n-15315 is mut twice.

### 启动命令
```shell
python3 -m oneflow.distributed.launch --nproc_per_node=1 motionrnn_base.py
```