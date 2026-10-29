# %%
import torch
import nnsight
from nnsight import TransformersModel

# %%
clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = (
    "After John and Mary went to the store, John gave a bottle of milk to"
)
model = TransformersModel("openai-community/gpt2", device_map="auto", dispatch=True)
# %%
correct_index = model.tokenizer(" John")["input_ids"][0]
incorrect_index = model.tokenizer(" Mary")["input_ids"][0]

print(correct_index)
print(incorrect_index)
# %%
import inspect

with model.trace(clean_prompt):
    print([m for m in inspect.getmembers(model)])
# %%
print(dir(model))
