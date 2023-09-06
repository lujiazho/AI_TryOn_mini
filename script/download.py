import gdown

url = 'https://drive.google.com/uc?id=1Zfj2r_4UePdcngHltlLD7bDquReBLPxh'
output = './stable_diffusion/models/sd/chilloutmix_NiPrunedFp16Fix.safetensors'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1kwIyzQBLRH5nsLbUaIKvmjPcoTlau0Io'
output = './stable_diffusion/models/sam/sam_vit_l_0b3195.safetensors'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1vBDYpoYxZCf0q4YFogKn4oVarocfnbZB'
output = './stable_diffusion/models/lora/girl_Mix_V40.safetensors'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1V1QlQhtlUr3pDfqf3SKDCK2YYXzYilET'
output = './stable_diffusion/models/cldm/diff_control_sd15_canny_fp16.safetensors'
gdown.download(url, output, quiet=False)