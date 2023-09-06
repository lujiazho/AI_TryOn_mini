# AI_TryOn_mini

<img src="assets/example/demo.png" width="100%"/>

# Tools

From top to bottom
- Clear image
- Drawer
- SAM point-segmenter with interactive functionality (left pos right neg)
- Rect-drawer for local area correction
- Undo
- Eraser
- Expand
- Download
- Send to render

# Run Locally

- Install the dependencies (if needed)
```shell
pip install -r requirements.txt
```
- Download models of [SAM](https://github.com/facebookresearch/segment-anything), [LDM](https://github.com/CompVis/stable-diffusion), [ControlNet](https://github.com/lllyasviel/ControlNet), and [LoRA](https://github.com/microsoft/LoRA).

You may download models from [civitai](https://civitai.com/) or
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget -P ./XMem/saves/ https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth
```
- Launch backend
```
python run_app.py
```
- Go to Browser
```
http://127.0.0.1:8001
```

# Hardware requirements

- At least 12G GPU memory is required
