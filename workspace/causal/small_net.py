# %%
from collections import OrderedDict

import torch
from nnsight import NNsight

# %%
# Create small net
input_size = 10
output_size = 1
hidden_dims = 10

net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(input_size, hidden_dims)),
            ("layer2", torch.nn.Linear(hidden_dims, output_size)),
        ]
    )
).requires_grad_(False)

input = torch.randn(1, input_size)
# %%
model = NNsight(net)
# %%
with model.trace(input):
    output = model.output
    print(output)
