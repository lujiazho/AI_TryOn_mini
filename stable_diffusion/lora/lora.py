import re
import torch

re_compiled = {}
re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}

class LoraUpDownModule:
    def __init__(self):
        self.up = None
        self.down = None
        self.alpha = None

def convert_diffusers_name_to_compvis(key):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key

def assign_lora_names_to_compvis_modules(model):
    lora_layer_mapping = {}

    for name, module in model.cond_stage_model.wrapped.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    for name, module in model.model.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    return lora_layer_mapping


def lora_forward(module, input, original_forward):
    res = original_forward(module, input)

    lora_layer_name = getattr(module, 'lora_layer_name', None)

    module = lora_modules.get(lora_layer_name, None)
    if module is None:
        return res

    res = res + module.up(module.down(input)) * 0.99 * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)

    return res

def lora_Linear_forward(self, input):
    return lora_forward(self, input, torch.nn.Linear_forward_before_lora)

def lora_Conv2d_forward(self, input):
    return lora_forward(self, input, torch.nn.Conv2d_forward_before_lora)

class PlugableLora(torch.nn.Module):
    def __init__(self, lora_sd, diffusion_model, device) -> None:
        super().__init__()
        
        self.lora_layer_mapping = assign_lora_names_to_compvis_modules(diffusion_model)
        lora_modules.clear()
        
        for key_diffusers in lora_sd:
            weight = lora_sd[key_diffusers]
            key_diffusers_without_lora_parts, lora_key = key_diffusers.split(".", 1)
            converted = convert_diffusers_name_to_compvis(key_diffusers_without_lora_parts)
            sd_module = self.lora_layer_mapping.get(converted)
            assert sd_module is not None, f"module {converted} not found"

            lora_module = lora_modules.get(converted, None)
            if lora_module is None:
                lora_module = LoraUpDownModule()
                lora_modules[converted] = lora_module

            if lora_key == 'alpha':
                lora_module.alpha = weight.item()
                continue

            if type(sd_module) == torch.nn.Linear:
                module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            elif type(sd_module) == torch.nn.Conv2d and weight.shape[2:] == (1, 1):
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
            else:
                print(f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}')

            with torch.no_grad():
                module.weight.copy_(weight)

            module.half().to(device)

            if lora_key == "lora_up.weight":
                lora_module.up = module
            elif lora_key == "lora_down.weight":
                lora_module.down = module
            else:
                print(f"lora key {lora_key} not supported")

        if not hasattr(torch.nn, 'Linear_forward_before_lora'):
            torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

        if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
            torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

    def hook(self):        
        torch.nn.Linear.forward = lora_Linear_forward
        torch.nn.Conv2d.forward = lora_Conv2d_forward

    def restore(self):
        torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
        torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora

lora_modules = {}