from .monai_unet import get_monai_unet
from .pytorch_unet import get_pytorch_unet

def get_model(name="monai", in_channels=4, out_channels=5):
    if name == "monai":
        return get_monai_unet(in_channels, out_channels)
    elif name == "pytorch":
        return get_pytorch_unet(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown model name: {name}")
