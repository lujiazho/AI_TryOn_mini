"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.cuda.amp import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything

import safetensors.torch
import torch.nn.functional as F

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from cldm.cldm import ControlParams, PlugableControlModel
from cldm.preprocessor import canny as apply_canny
from cldm.preprocessor import detectmap_proc

from lora.lora import PlugableLora
from CLIP import clip

import torch
from PIL import Image

def save_tensor_as_grid(tensor, filename):
    # Ensure the tensor is CPU based
    tensor = tensor.cpu()
    # map the tensor to range [0, 1]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # Split the tensor into separate channels
    channels = torch.split(tensor, 1, dim=1)
    
    # Convert each channel to a grayscale image and store in a list
    images = [Image.fromarray((255 * c.squeeze().numpy()).astype('uint8')) for c in channels]
    
    # Create an empty image with the right size to hold the 2x2 grid
    width, height = images[0].size
    grid_image = Image.new('L', (2 * width, 2 * height))
    
    # Paste each image into the grid
    grid_image.paste(images[0], (0, 0))
    grid_image.paste(images[1], (width, 0))
    grid_image.paste(images[2], (0, height))
    try:
        grid_image.paste(images[3], (width, height))
    except IndexError:
        pass
    
    # Save the grid image as a PNG
    grid_image.save(filename)

def save_tensor_as_image(tensor, filename):
    # Ensure tensor is on the CPU and detach it from gradients if needed
    tensor = tensor.cpu().detach()
    # map the tensor to range [0, 1]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # Convert tensor values to the range [0, 255] and change data type
    tensor = (tensor.squeeze() * 255).byte()
    
    # Convert tensor to numpy array and transpose dimensions for PIL Image
    array = tensor.numpy().transpose(1, 2, 0)
    
    # Create an image from the array
    img = Image.fromarray(array, 'RGB')
    
    # Save the image
    img.save(filename)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    try:
        pl_sd = torch.load(ckpt, map_location="cpu")
    except:
        pl_sd = safetensors.torch.load_file(ckpt, device="cpu")

    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    model = instantiate_from_config(config.model)
    try:
        sd = pl_sd["state_dict"]
    except KeyError:
        sd = pl_sd
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def standardized_resize(img):
    w, h = img.size
    # make w, h not more than 768, resize with same aspect ratio
    if w > 768 or h > 768:
        if w > h:
            h = int(h * 768 / w)
            w = 768
        else:
            w = int(w * 768 / h)
            h = 768
    w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 32
    img = img.resize((w, h), resample=PIL.Image.LANCZOS)
    return img

def load_img(path):
    try:
        image = Image.open(path).convert("RGB")
        print(f"loaded input image of size ({image.width}, {image.height}) from {path}")
    except:
        image = Image.fromarray(path).convert("RGB")
    image = standardized_resize(image)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

MODEL = None
LORA = None
CONTROL_NET = None

def render_image(prompt, init_img, mask, outdir, skip_grid, skip_save, ddim_steps, ddim_eta, n_iter, n_samples, n_rows, scale, strength, config, cldm_config, ckpt, sideunet_safetensors, lora_safetensors, seed, precision, rect=None, control_img=None):
    global MODEL, LORA, CONTROL_NET

    seed = seed_everything(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load difusion model
    if MODEL is None:
        config = OmegaConf.load(f"{config}")
        model = load_model_from_config(config, f"{ckpt}")
        model = model.half().to("cpu")
        # hijack clip model with custom forward
        model_hijack = clip.StableDiffusionModelHijack()
        model_hijack.hijack(model, device)
        MODEL = model.to("cpu")
    else:
        model = MODEL.to("cpu")

    # load lora
    if LORA is None:
        lora_sd = safetensors.torch.load_file(lora_safetensors, device="cpu")
        LORA = lora_sd
    else:
        lora_sd = LORA
    plugable_lora = PlugableLora(lora_sd, model, device)
    plugable_lora.hook()

    # load controlnet
    if CONTROL_NET is None:
        sideunet_params = safetensors.torch.load_file(sideunet_safetensors, device="cpu")
        CONTROL_NET = sideunet_params
    else:
        sideunet_params = CONTROL_NET

    # hook controlnet to diffusion model
    control_img = control_img if control_img is not None else init_img
    try:
        img = Image.open(control_img).convert("RGB")
    except:
        # treat as numpy array
        img = Image.fromarray(control_img).convert("RGB")
    img = np.array(standardized_resize(img))
    if rect is not None:
        start_x, start_y, end_x, end_y = rect['start_x'], rect['start_y'], rect['end_x'], rect['end_y']
        img = img[start_y:end_y, start_x:end_x, :]
    canny_map = apply_canny(img, thr_a=100, thr_b=200)
    H, W = canny_map.shape[0], canny_map.shape[1]
    canny_input, canny_vis_map = detectmap_proc(canny_map, "canny", False, H, W, device)
    plugable_control_model = PlugableControlModel(
        state_dict=sideunet_params, 
        config_path=cldm_config, 
        base_model=model,
        control_param=ControlParams(
            hint_cond=canny_input,
            weight=1.0,
            guidance_stopped=False, # for controlling guidance steps/percents
            start_guidance_percent=0,
            stop_guidance_percent=1
        ),
        plugable_lora=None
    )
    plugable_control_model.control_model = plugable_control_model.control_model.half().to("cpu")
    plugable_control_model.hook(model.model.diffusion_model)
    
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size

    assert prompt is not None
    data = [batch_size * [prompt]]

    if not skip_grid or not skip_save:
        os.makedirs(outdir, exist_ok=True)
        outpath = outdir

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

    init_image = load_img(init_img).half().to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    if rect is not None:
        cropped_init_image = init_image[:, :, start_y:end_y, start_x:end_x]
    print("init_image.shape", init_image.shape)

    mask = load_img(mask).half().to(device) # after load_img, it'll be in [-1, 1]
    mask = (mask + 1.0) / 2.0 # now in [0, 1]
    idx = mask < 0.1
    mask[idx] = 1
    mask[~idx] = 0
    mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    if rect is not None:
        cropped_mask = mask[:, :, start_y:end_y, start_x:end_x]
    
    # masked latent to be original
    model.first_stage_model = model.first_stage_model.to(device)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
    model.first_stage_model = model.first_stage_model.to("cpu")
    if rect is not None:
        latent_start_x, latent_start_y, latent_end_x, latent_end_y = start_x // 8, start_y // 8, end_x // 8, end_y // 8
        cropped_init_latent = init_latent[:, :, latent_start_y:latent_end_y, latent_start_x:latent_end_x]
    # restored_image = model.decode_first_stage(init_latent)
    # save_tensor_as_image(restored_image, './restored_image.png')
    # save_tensor_as_grid(init_latent, './init_latent.png')

    # Resize the mask tensor
    mask_latent = F.interpolate(mask[:, 0:1, ...], size=(init_latent.shape[2], init_latent.shape[3]))
    mask_latent = mask_latent.repeat(1, 4, 1, 1)
    if rect is not None:
        cropped_mask_latent = mask_latent[:, :, latent_start_y:latent_end_y, latent_start_x:latent_end_x]
    # save_tensor_as_grid(mask_latent, './mask_latent.png')

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope(enabled=True):
            # move to [0, 1] for later visualization
            init_image = (init_image + 1) / 2.0
            if rect is not None:
                cropped_init_image = (cropped_init_image + 1) / 2.0

            all_samples = list()
            for n in trange(n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    model.cond_stage_model = model.cond_stage_model.to(device)
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * ["distorted face, Deformed, extra limbs, extra eyes, extra noses, extra ears, ugly, paintings, sketches, (worst quality:2), (low quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, nsfw, nipples"])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    model.cond_stage_model = model.cond_stage_model.to("cpu")

                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                    if rect is not None:
                        z_enc = init_latent.clone()
                        cropped_z_enc = z_enc[:, :, latent_start_y:latent_end_y, latent_start_x:latent_end_x]
                    # save_tensor_as_grid(z_enc, './z_enc.png')
                    # decode it
                    if rect is not None:
                        samples = sampler.decode(cropped_z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, mask=cropped_mask_latent, init_latent=cropped_init_latent, 
                                                callback=plugable_control_model.guidance_schedule_handler)
                    else:
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, mask=mask_latent, init_latent=init_latent, 
                                                callback=plugable_control_model.guidance_schedule_handler)

                    model.first_stage_model = model.first_stage_model.to(device)
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    model.first_stage_model = model.first_stage_model.to("cpu")

                    if not skip_save:
                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1
                    if rect is not None:
                        if x_samples.shape[2] != cropped_mask.shape[2] or x_samples.shape[3] != cropped_mask.shape[3]:
                            x_samples = F.interpolate(x_samples, size=(cropped_mask.shape[2], cropped_mask.shape[3]))
                        x_samples = x_samples * cropped_mask + cropped_init_image * (1 - cropped_mask)
                    else:
                        x_samples = x_samples * mask + init_image * (1 - mask)
                    all_samples.append(x_samples)

            if not skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

    plugable_control_model.restore(model.model.diffusion_model)
    plugable_lora.restore()

    MODEL = MODEL.to("cpu")
    plugable_control_model.control_model = plugable_control_model.control_model.to("cpu")

    torch.cuda.empty_cache()

    # return the first sample and it's second channel (the first is uc)
    ret_sample = (rearrange(all_samples[0][1].cpu().numpy(), 'c h w -> h w c') * 255).astype(np.uint8)
    return ret_sample, canny_vis_map, seed

def main():
    # python -m scripts.control_inpaint --prompt "A beautiful girl, sitting on bench" --init-img data/inpainting_examples/bench2.png --mask data/inpainting_examples/bench2_mask.png --strength 0.9 --n_samples 1 --ckpt /ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/stable/tmp/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp16Fix.safetensors
    # python -m scripts.control_inpaint --prompt "A beautiful slim girl standing by a grid bar" --init-img data/inpainting_examples/6458524847_2f4c361183_k.png --mask data/inpainting_examples/6458524847_2f4c361183_k_mask.png --strength 0.9 --n_samples 1 --ckpt /ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/stable/tmp/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp16Fix.safetensors
    # python -m scripts.control_inpaint --prompt "best quality, ultra high res, (photorealistic:1.4), 1woman, sleeveless white button shirt, black skirt, black choker, cute, (Kpop idol), (aegyo sal:1), (platinum blonde hair:1), ((puffy eyes)), looking at viewer, full body, facing front" --init-img data/inpainting_examples/2.png --mask data/inpainting_examples/2_mask.png --n_samples 1 --ckpt /ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/stable/tmp/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp16Fix.safetensors
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--mask",
        type=str,
        nargs="?",
        help="path to the mask image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/inpaint-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=28,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--cldm_config",
        type=str,
        default="configs/cldm/cldm_v15.yaml",
        help="path to config which constructs control model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--sideunet_safetensors",
        type=str,
        default="models/cldm/stable-diffusion-v1/diff_control_sd15_canny_fp16.safetensors",
        help="path to checkpoint of sideunet",
    )
    parser.add_argument(
        "--lora_safetensors",
        type=str,
        default="lora/girl_Mix_V40.safetensors", # girl_Mix_V40, koreanDollLikeness_v10
        help="path to checkpoint of lora",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    prompt, init_img, mask, outdir, skip_grid, skip_save, ddim_steps, ddim_eta, n_iter, n_samples, n_rows, scale, strength, config, cldm_config, ckpt, sideunet_safetensors, lora_safetensors, seed, precision = opt.prompt, opt.init_img, opt.mask, opt.outdir, opt.skip_grid, opt.skip_save, opt.ddim_steps, opt.ddim_eta, opt.n_iter, opt.n_samples, opt.n_rows, opt.scale, opt.strength, opt.config, opt.cldm_config, opt.ckpt, opt.sideunet_safetensors, opt.lora_safetensors, opt.seed, opt.precision
    render_image(prompt, init_img, mask, outdir, skip_grid, skip_save, ddim_steps, ddim_eta, n_iter, n_samples, n_rows, scale, strength, config, cldm_config, ckpt, sideunet_safetensors, lora_safetensors, seed, precision)

    print(f"Enjoy.")


if __name__ == "__main__":
    main()
