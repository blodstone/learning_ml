# %%
import nnsight
import torch
from nnsight import TransformersModel

# %%
clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = (
    "After John and Mary went to the store, John gave a bottle of milk to"
)
model = TransformersModel("openai-community/gpt2", device_map="auto", dispatch=True)
# %%

with torch.no_grad(), model.trace() as tracer:
    pass
