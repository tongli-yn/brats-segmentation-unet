from monai.networks.nets import UNet

def get_monai_unet(in_channels=4, out_channels=5):
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm='batch'
    )
    return model