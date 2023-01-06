import oneflow as flow
import oneflow.nn as nn

P01 = flow.placement(type="cuda", ranks=[0, 1])
P23 = flow.placement(type="cuda", ranks=[2, 3])

"""
python3 -m oneflow.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=2 --master_addr=master --master_port=7788 main.py
"""

class StageModule(nn.Module):
    def __init__(self, in_dims, out_dims, placement=None, sbp=None):
        super().__init__()
        self.w = nn.Parameter(
            flow.randn(in_dims, out_dims, placement=placement, sbp=sbp)
        )

    def forward(self, x):
        out = flow.matmul(x, self.w)
        return out


class ModuleModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 模型第一阶段在第 0 和第 1 卡上进行数据并行计算
        self.m_stage0 = StageModule(5, 8, placement=P01, sbp=flow.sbp.broadcast)

        # 模型第二阶段在第 2 和第 3 卡上进行数据并行计算
        self.m_stage1 = StageModule(8, 3, placement=P23, sbp=flow.sbp.broadcast)

    def forward(self, x):
        # 第一阶段，数据切分在第 0 和第 1 卡，用于数据并行
        out_stage0 = self.m_stage0(x)

        # 第二阶段需要将输入数据还原完整，并转移至第 2 和第 3 卡，用于模型并行
        in_stage1 = out_stage0.to_global(placement=P23, sbp=flow.sbp.split(dim=0))
        out_stage1 = self.m_stage1(in_stage1)

        return out_stage0, out_stage1


# Graph
class GraphModel(nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = ModuleModel()
        self.model.m_stage0.to(nn.graph.GraphModule).set_stage(stage_id=0, placement=P01)
        self.model.m_stage1.to(nn.graph.GraphModule).set_stage(stage_id=1, placement=P23)

    def build(self, x):
        return self.model(x)


if __name__ == "__main__":
    graph = GraphModel()
    # 需要将输入数据切分，用于数据并行
    in_stage0 = flow.randn(4, 5, placement=P01, sbp=flow.sbp.split(dim=0))  # 实际上是两个2*5
    out_stage0, out_stage1 = graph(in_stage0)
    print(out_stage0.shape, out_stage1.shape)  # (4, 8) (4, 3)
