import os

cmd1 = f'python3 -m oneflow.distributed.launch --nproc_per_node=2 test03.py'
cmd2 = f'python3 -m oneflow.distributed.launch --nproc_per_node=2 test03.py'

os.system(cmd1)
os.system(cmd2)
