# %%
import plotly.express as px
import torch
import tqdm
from nnsight import TransformersModel
import rootutils

path = rootutils.find_root(search_from='.', indicator='pyproject.toml')
# %%
# Model loading and setup input

model = TransformersModel(
    'google/gemma-3-1b-pt',
    device_map='auto',
    dispatch=True,
)

# %%
clean_prompt = 'Paris is in the country of'
corrupted_prompt = 'Berlin is in the country of'

correct_ans_idx = model.tokenizer('France')['input_ids'][1]
incorrect_ans_idx = model.tokenizer('Germany')['input_ids'][1]
# %%
with model.generate(corrupted_prompt, max_new_tokens=1, temperature=1e-5) as tracer:
    ids = tracer.result.save()
print(model.tokenizer.decode(ids))
# %%
print(correct_ans_idx)
print(model.tokenizer('France')['input_ids'])
# %%
N_LAYERS = len(model.model.layers)
h_layers_clean = []
h_layers_corrupt = []
# Get Total Effect
# Corrupted prompt should have lower logit for the correct answer than the clean prompt
# The logit_diff corrupt should be lower than the logit_diff clean
with model.trace() as tracer:
    with tracer.invoke(corrupted_prompt):
        for i in range(N_LAYERS):
            h_layers_corrupt.append(model.model.layers[i].output.save())
        logits_corrupt = model.lm_head.output.save()
        logits_diff_corrupt = (
            logits_corrupt[0][-1, correct_ans_idx]
            - logits_corrupt[0][-1, incorrect_ans_idx]
        )
        logits_diff_corrupt.save()
    with tracer.invoke(clean_prompt):
        for i in range(N_LAYERS):
            h_layers_clean.append(model.model.layers[i].output.save())
        logits_clean = model.lm_head.output.save()
        logits_diff_clean = (
            logits_clean[0][-1, correct_ans_idx]
            - logits_clean[0][-1, incorrect_ans_idx]
        )
        logits_diff_clean.save()
# This is negative treatment effect, therefore the total effect is negative
print('Logit diff clean:', logits_diff_clean)
print('Logit diff corrupt:', logits_diff_corrupt)
total_effect = logits_diff_corrupt - logits_diff_clean
print(total_effect)
# %%
# Calculate the indirect effect of each layer
# Activation patching
indirect_effects = []
with model.trace(corrupted_prompt):
    corrupt_input_ids = model.model.input.save()
with model.trace() as tracer:
    for i in tqdm.tqdm(range(N_LAYERS)):
        _indirect_effects = []
        for token_idx in range(len(corrupt_input_ids[0])):
            with tracer.invoke(corrupted_prompt):
                model.model.layers[i].output[:, token_idx, :] = h_layers_clean[i][
                    :, token_idx, :
                ]
                logits_patched = model.lm_head.output.save()
                logits_diff_patched = (
                    logits_patched[0][-1, correct_ans_idx]
                    - logits_patched[0][-1, incorrect_ans_idx]
                )
                # Since patching a clean activation is a positive treatment effect, the indirect effect is positive
                ie = logits_diff_patched.cpu() - logits_diff_corrupt.cpu()
                _indirect_effects.append(ie.detach().float().item())
        indirect_effects.append(_indirect_effects)
fig = px.imshow(
    indirect_effects,
)
fig.write_html(path / 'plotly/indirect_effects_gemma.html')
# %%
# Let's see how the prediction change when we patch the activation of a specific token in a specific layer
layer = 16  # the highest indirect effect layer
token_idx = 0
with model.trace(corrupted_prompt):
    # first see the output of the model with the corrupted prompt
    model.model.layers[layer].output[:, token_idx, :] = h_layers_clean[layer][
        :, token_idx, :
    ]
    logits_corrupt = model.lm_head.output
    # get the top one
    prediction = torch.argmax(logits_corrupt[0][-1, :]).save()
print(model.tokenizer.decode(prediction))  # The answer should be "France" now


# %%
# Attribution patching
def metric(logits, correct_ans_idx, incorrect_ans_ids):
    return logits[0][-1, correct_ans_idx] - logits[0][-1, incorrect_ans_ids]


indirect_effects_attr = []
with model.trace(corrupted_prompt):
    all_layers = []
    for i in range(N_LAYERS):
        component_corrupted = model.model.layers[i].output.save()
        all_layers.append(component_corrupted)
    logits_corrupted = model.lm_head.output.save()
    logits_diff_corrupted = metric(logits_corrupted, correct_ans_idx, incorrect_ans_idx)
    with logits_diff_corrupted.backward():
        for j in reversed(range(N_LAYERS)):
            _indirect_effects_attr = []
            component_grad = all_layers[j].grad
            activation_diff = h_layers_clean[j] - h_layers_corrupt[j]
            # [1, T, D] @ [1, T, D] -> [1, T, 1]
            ie = torch.einsum(
                'ijk, ijk->ij', component_grad.cpu(), activation_diff.cpu()
            )
            for token_idx in range(len(corrupt_input_ids[0])):
                _indirect_effects_attr.append(ie[:, token_idx].float().item())
            indirect_effects_attr.append(_indirect_effects_attr)
fig = px.imshow(
    indirect_effects_attr[::-1],
)
fig.write_html(path / 'plotly/indirect_effects_attr_gemma.html')
# %%
