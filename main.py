import oneflow as flow

BROADCAST = [flow.sbp.broadcast]
P0 = flow.placement("cuda", ranks=[0])

x, y = flow.meshgrid(
            flow.arange(-(3 - 1) // 2, (3 - 1) // 2 + 1),
            flow.arange(-(3 - 1) // 2, (3 - 1) // 2 + 1))




