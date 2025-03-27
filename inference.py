import torch
from torch.utils.data import DataLoader
from datasets.brats_dataset import BraTSDataset2D
from utils.visualizer import visualize_prediction
from monai.transforms import Compose, Resize, ScaleIntensityRange, ToTensor
import pandas as pd
from models import get_model

if __name__ == "__main__":
    model_name = "monai"
    in_channels = 4
    out_channels = 5
    image_size = (128, 128)

    model = get_model(model_name, in_channels, out_channels)
    model.load_state_dict(torch.load(f"best_{model_name}_unet.pth"))
    model.eval()
    model.cuda()

    df = pd.read_csv("data/meta_data.csv")

    image_transform = Compose([
        ScaleIntensityRange(0, 1, 0.0, 1.0, clip=True),
        Resize(image_size),
        ToTensor()
    ])
    label_transform = Compose([
        Resize(image_size, mode="nearest"),
        ToTensor()
    ])

    dataset = BraTSDataset2D(df, image_transform, label_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for image, label in loader:
            image = image.cuda()
            pred = torch.argmax(model(image), dim=1)
            visualize_prediction(image[0].cpu(), label[0], pred[0])
            break  # only show one for demo
