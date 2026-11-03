# %%
import pprint

import torch
from nnsight import TransformersModel

# %%
# Model loading and setup input

model = TransformersModel(
    'Qwen/Qwen3.5-2B-Base',
    device_map='auto',
    dispatch=True,
)


clean_prompt = 'Paris is in the country of '
corrupted_prompt = 'Rio is in the country of '

correct_ans_idx = model.tokenizer('France')['input_ids'][0]
incorrect_ans_idx = model.tokenizer('Brazil')['input_ids'][0]
# %%

N_LAYERS = len(model.model.language_model.layers)

# Get Total Effect
with model.trace() as tracer:  # noqa: SIM117
    with tracer.invoke(corrupted_prompt):
        logits = model.lm_head.output.save()
        logits_diff_corrupt = (
            logits[0][-1, correct_ans_idx] - logits[0][-1, incorrect_ans_idx]
        )
        logits_diff_corrupt.save()
    with tracer.invoke(clean_prompt):
        logits = model.lm_head.output.save()
        logits_diff_clean = (
            logits[0][-1, correct_ans_idx] - logits[0][-1, incorrect_ans_idx]
        )
        logits_diff_clean.save()
total_effect = logits_diff_corrupt - logits_diff_clean
# %%
print(total_effect)
