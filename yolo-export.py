"""
YOLO to ONNX Export Utility
--------------------------
Converts Ultralytics YOLO models (.pt) to ONNX format optimized for
Apple Silicon (M-series) inference.

Settings used:
- Fixed input shape (dynamic=False) for better Neural Engine acceleration.
- Simplification enabled to fuse layers for lower latency.
- FP32 precision maintained for maximum compatibility with CoreML providers.
- (default) Input shape: (1, 3, 640, 640) BCHW
- Output shape: (1, 7, 8400) --> 4 box coordinates + 3 labels confidence scores (robot, target, and obstacle,
respectively)
- Size (memory): 48 MB
- Size (parameters): 12,749,288
"""  # noqa

from argparse import ArgumentParser, Namespace

from ultralytics import YOLOWorld


def parse_args() -> Namespace:
    def str_to_list(s: str) -> list[str]:
        return s.split(", ")

    args = ArgumentParser()

    args.add_argument(
        "--src",
        type=str,
        default="assets/models/yolo/yolov8s-worldv2.pt",
        help="Path to the original .pt model file.",
    )

    args.add_argument(
        "--dest",
        type=str,
        default="assets/models/yolo",
        help="Exported .onnx model destination.",
    )

    args.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (square).",
    )

    args.add_argument(
        "--labels",
        type=str_to_list,
        default=[
            "robot with rubber tracks",
            "yellow small circular block",
            "white cardboard box",
        ],
        help=(
            "Comma-separated prompts describing target objects the model needs "
            "to detect."
        ),
    )

    return args.parse_args()


if __name__ == "__main__":
    arg = parse_args()

    model = YOLOWorld(arg.src)

    model.set_classes(arg.labels)

    print(f"Starting export for {arg.src}...")

    path = model.export(
        format="onnx",
        imgsz=[arg.imgsz, arg.imgsz],
        simplify=True,  # cleaning up the ONNX graph for better performance
        opset=12,  # standard version for compatibility
        dynamic=False,  # fixed input shape for faster inference
        half=False,  # keeping FP32
    )

    print(f"Export successful. File saved at: {path}")
