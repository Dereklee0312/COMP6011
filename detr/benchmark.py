import json
import time
from pathlib import Path

import torch
from ultralytics import RTDETR
from ultralytics.utils import YAML


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CONFIG = PROJECT_DIR / "config.yaml"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "benchmark_results"
DEFAULT_CHECKPOINT = "./train/weights/best.pt"


def resolve_output_path(checkpoint: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    DEFAULT_RESULTS_DIR.mkdir(exist_ok=True)
    return DEFAULT_RESULTS_DIR / f"{checkpoint.stem}_benchmark.json"


def load_data_config(data_config: Path) -> dict:
    return YAML.load(data_config)


def load_validation_images(data: dict) -> list[Path]:
    dataset_root = Path(data["path"])
    val_dir = dataset_root / data["val"]
    return sorted(path for path in val_dir.iterdir() if path.is_file())


def synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def time_inference(
    model: RTDETR,
    image_paths: list[Path],
    imgsz: int,
    device: str,
    conf: float,
    warmup_images: int,
) -> tuple[float, float]:
    warmup = image_paths[: min(warmup_images, len(image_paths))]
    measured = image_paths[min(warmup_images, len(image_paths)) :] or image_paths

    for image_path in warmup:
        model.predict(source=str(image_path), imgsz=imgsz, device=device, conf=conf, verbose=False)

    latencies_ms: list[float] = []
    for image_path in measured:
        synchronize_device(device)
        start = time.perf_counter()
        model.predict(source=str(image_path), imgsz=imgsz, device=device, conf=conf, verbose=False)
        synchronize_device(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
    fps = 1000.0 / mean_latency_ms
    return mean_latency_ms, fps

class Config:
    def __init__(self):
        self.checkpoint = DEFAULT_CHECKPOINT
        self.data = DEFAULT_DATA_CONFIG
        self.imgsz = 640
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.conf = 0.25
        self.warmup_images = 20
        self.max_images = None
        self.output = None


def main() -> None:
    args = Config()
    checkpoint = Path(args.checkpoint).resolve()
    data_config = Path(args.data).resolve()
    output_path = resolve_output_path(checkpoint, args.output.resolve() if args.output else None)
    data = load_data_config(data_config)

    model = RTDETR(str(checkpoint))
    validation_metrics = model.val(data=str(data_config), imgsz=args.imgsz, device=args.device, verbose=False)
    image_paths = load_validation_images(data)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    mean_latency_ms, fps = time_inference(
        model=model,
        image_paths=image_paths,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        warmup_images=args.warmup_images,
    )

    output = {
        "checkpoint": str(checkpoint),
        "data_config": str(data_config),
        "dataset_path": data["path"],
        "validation_images": len(image_paths),
        "imgsz": args.imgsz,
        "device": args.device,
        "confidence_threshold": args.conf,
        "metrics": validation_metrics.results_dict,
        "mean_latency_ms": mean_latency_ms,
        "fps": fps,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
