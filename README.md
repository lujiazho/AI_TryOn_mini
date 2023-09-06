# AI_TryOn_mini

<table>
    <tr>
        <td align="center">
        <img src="assets/example/demo.png" width="240"/>
        </td>
    </tr>
</table>

# Demo

<img src="example/demo.gif" width="240" />

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
(Preparing)
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

# TODO

- CodeFormer
- Memory usage optimization
- ...