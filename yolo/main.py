# Code from: https://docs.ultralytics.com/datasets/detect/kitti/

import torch
from ultralytics import YOLO
def main():
    # Load a model
    model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="./config.yaml", epochs=10, imgsz=640, device="mps")


if __name__ == "__main__":
    main()
