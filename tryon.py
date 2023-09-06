# uvicorn tryon:app --host 0.0.0.0
import os
import sys
absolute_path = os.path.abspath('./stable_diffusion')
print(absolute_path)
sys.path.append(absolute_path)

import numpy as np
from io import BytesIO
from base64 import b64encode, b64decode
from PIL import Image, PngImagePlugin
from script import render_image

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

params_dict = {
    "prompt": "best quality, detailed face, ((masterpiece)), high res, extremely detailed, intricate detail, hair, fine detail, (photorealistic:1.4), 1girl, cute, (Kpop idol), ((puffy eyes)), natural light, smile",
    "init_img": None,
    "mask": None,
    "outdir": "", # not used
    "skip_grid": True,
    "skip_save": True,
    "ddim_steps": 28,
    "ddim_eta": 0.0,
    "n_iter": 1,
    "n_samples": 2,
    "n_rows": 0,
    "scale": 7.5,
    "strength": 0.75,
    "config": "stable_diffusion/configs/v1-inference.yaml",
    "cldm_config": "stable_diffusion/configs/cldm_v15.yaml",
    "ckpt": "stable_diffusion/models/sd/chilloutmix_NiPrunedFp16Fix.safetensors",
    "sideunet_safetensors": "stable_diffusion/models/cldm/diff_control_sd15_canny_fp16.safetensors",
    "lora_safetensors": "stable_diffusion/models/lora/girl_Mix_V40.safetensors", # koreanDollLikeness_v10, girl_Mix_V40
    "seed": -1,
    "precision": "autocast",
    "rect": None
}

import torch
from segment_anything import sam_model_registry, SamPredictor#, SamAutomaticMaskGenerator

sam_checkpoint = "./stable_diffusion/models/sam/sam_vit_l_0b3195.pth" # "sam_vit_l_0b3195.pth" or "sam_vit_h_4b8939.pth"
model_type = "vit_l" # "vit_l" or "vit_h"
if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    device = 'cpu'
cpu_device = torch.device('cpu')

print("Loading model")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
print("Finishing loading")
predictor = SamPredictor(sam)

segmented_mask = []
interactive_mask = []
mask_input = None

SEED = None
GLOBAL_IMAGE = None
GLOBAL_MASK = None
GLOBAL_RENDERED = None

from fastapi import FastAPI, status, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_main():
    return read_content('ui.html')

@app.get("/assets/{path}/{file_name}", response_class=FileResponse)
async def read_assets(path, file_name):
    return f"assets/{path}/{file_name}"

@app.post("/controlnet/img2img")
async def process_images(
    # preprocessor: str = Form(...),
    # weight: float = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    # resizeMode: str = Form(...),
    # processorRes: int = Form(...),
    # thresholdA: float = Form(...),
    # thresholdB: int = Form(...),
    # guidanceStart: float = Form(...),
    # guidanceEnd: float = Form(...),
    # guidance: float = Form(...)
):
    global SEED, GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_RENDERED, sam

    sam = sam.to(cpu_device)

    # Read the image and mask data as bytes
    image_data = await image.read()
    mask_data = await mask.read()

    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data))
    if img.shape[-1] == 4:
        GLOBAL_IMAGE = img[:,:,:-1]
    else:
        GLOBAL_IMAGE = img

    mask_data = BytesIO(mask_data)
    mask_img = np.array(Image.open(mask_data))[...,-1]
    GLOBAL_MASK = mask_img

    params_dict.update({
        "init_img": img,
        "control_img": img,
        "mask": mask_img,
        "seed": -1,
        "rect": None,
        "ddim_steps": 28,
        "strength": 0.75,
    })

    rendered_, control_, seed = render_image(**params_dict)
    if type(rendered_) is list:
        rendered_ = rendered_[0]

    GLOBAL_RENDERED = rendered_

    print("rendered", rendered_.shape)
    print("control", control_.shape)

    rendered_, control_ = Image.fromarray(rendered_), Image.fromarray(control_)
    control_.save("control.png")
    SEED = seed

    render_base64 = pil_image_to_base64(rendered_)
    control_base64 = pil_image_to_base64(control_)

    sam.to(device)
    
    # Return a JSON response
    return JSONResponse(
        content={
            # "preprocessor": preprocessor,
            # "resizeMode": resizeMode,
            # "weight": weight,
            # "guidance": guidance,
            # "guidanceStart": guidanceStart,
            # "guidanceEnd": guidanceEnd,
            # "processorRes": processorRes,
            # "thresholdA": thresholdA,
            # "thresholdB": thresholdB,
            "render": render_base64,
            "control": control_base64,
            "message": "Images received successfully",
        },
        status_code=200,
    )

@app.post("/uploadimage")
async def process_images(
    image: UploadFile = File(...)
):
    global segmented_mask, interactive_mask
    global GLOBAL_IMAGE, SEED, GLOBAL_MASK

    SEED = None
    GLOBAL_MASK = None
    GLOBAL_RENDERED = None

    segmented_mask = []
    interactive_mask = []

    # Read the image and mask data as bytes
    image_data = await image.read()

    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data))
    print("get image", img.shape)
    GLOBAL_IMAGE = img[:,:,:-1]

    with torch.no_grad():
        predictor.set_image(GLOBAL_IMAGE)

    torch.cuda.empty_cache()
    
    print("finish setting image")
    # Return a JSON response
    return JSONResponse(
        content={
            "message": "Images received successfully",
        },
        status_code=200,
    )

from fastapi import Request

@app.post("/segmentation/click")
async def click_images(
    request: Request,
):  
    global mask_input, interactive_mask
    
    form_data = await request.form()
    type_list = [int(i) for i in form_data.get("type").split(',')]
    click_list = [int(i) for i in form_data.get("click_list").split(',')]

    point_coords = np.array(click_list, np.float32).reshape(-1, 2)
    point_labels = np.array(type_list).reshape(-1)

    print(point_coords)
    print(point_labels)

    if (len(point_coords) == 1):
        mask_input = None
    
    with torch.no_grad():
        masks_, scores_, logits_ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=True,
        )

    best_idx = np.argmax(scores_)
    res = masks_[best_idx]
    mask_input = logits_[best_idx][None, :, :]

    len_prompt = len(point_labels)
    len_mask = len(interactive_mask)
    if len_mask == 0 or len_mask < len_prompt:
        interactive_mask.append(res)
    else:
        interactive_mask[len_prompt-1] = res

    # Return a JSON response
    res = Image.fromarray(res)

    torch.cuda.empty_cache()
    
    # Return a JSON response
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res),
            "message": "Images processed successfully"
        },
        status_code=200,
    )

@app.post("/finish_click")
async def finish_interactive_click(
    mask_idx: int = Form(...),
):
    global segmented_mask, interactive_mask

    segmented_mask.append(interactive_mask[mask_idx])
    interactive_mask = list()

    return JSONResponse(
        content={
            "message": "Finish successfully",
        },
        status_code=200,
    )

@app.post("/undo")
async def undo_mask():
    global segmented_mask

    segmented_mask.pop()

    return JSONResponse(
        content={
            "message": "Clear successfully",
        },
        status_code=200,
    )

@app.post("/rect")
async def rect_images(
    start_x: int = Form(...), # horizontal
    start_y: int = Form(...), # vertical
    end_x: int = Form(...), # horizontal
    end_y: int = Form(...)  # vertical
):

    # crop image within the box
    global GLOBAL_IMAGE, SEED, GLOBAL_MASK, sam
    assert SEED is not None, "Please upload an image first"

    sam = sam.to(cpu_device)

    # params_dict.update({
    #     "init_img": GLOBAL_IMAGE,
    #     "mask": GLOBAL_MASK,
    #     "seed": SEED,
    #     "rect": {
    #         "start_x": start_x,
    #         "start_y": start_y,
    #         "end_x": end_x,
    #         "end_y": end_y,
    #     },
    #     "ddim_steps": 4,
    # })
    cur_mask = np.ones_like(GLOBAL_MASK) * 255
    # copy value of GLOBAL_MASK to cur_mask
    cur_mask[start_y:end_y, start_x:end_x] = GLOBAL_MASK[start_y:end_y, start_x:end_x]
    params_dict.update({
        "init_img": GLOBAL_RENDERED,
        "control_img": GLOBAL_IMAGE,
        "mask": cur_mask,
        "seed": -1,
        "rect": None,
        "ddim_steps": 28,
    })

    rendered_, control_, seed = render_image(**params_dict)
    # assert seed == SEED, "seed should be the same"

    if type(rendered_) is list:
        rendered_ = rendered_[0]

    rendered_ = rendered_[start_y:end_y, start_x:end_x]
    control_ = control_[start_y:end_y, start_x:end_x]

    print("rendered", rendered_.shape)
    print("control", control_.shape)

    rendered_, control_ = Image.fromarray(rendered_), Image.fromarray(control_)

    xray_base64 = pil_image_to_base64(rendered_)
    control_base64 = pil_image_to_base64(control_)

    sam = sam.to(device)

    # Return a JSON response
    return JSONResponse(
        content={
            "masks": xray_base64,
            "control": control_base64,
            "message": "Images processed successfully"
        },
        status_code=200,
    )