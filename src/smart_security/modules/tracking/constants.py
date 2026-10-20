# 模块：跟踪常量
# 作用：集中 COCO 类别、画框颜色与检测行格式，避免魔法数字散落。

from __future__ import annotations

PERSON_CLASS_ID = 0
TRACK_ROW_MIN_LEN = 7
DETECTION_ROW_MIN_LEN = 6
FPS_BUFFER_SIZE = 30
QUIT_KEY = ord("q")
SCREENSHOT_KEY = ord("s")

COCO_CLASS_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

VEHICLE_CLASS_NAMES: frozenset[str] = frozenset(
    {"bicycle", "car", "motorcycle", "bus", "truck"}
)
ANIMAL_CLASS_NAMES: frozenset[str] = frozenset(
    {"bird", "cat", "dog", "horse", "sheep", "cow"}
)

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (0, 165, 255),
    "vehicle": (255, 0, 0),
    "animal": (0, 255, 0),
    "other": (255, 255, 0),
}

ZONE_FILL_COLOR = (0, 255, 0)
ZONE_OVERLAY_ALPHA = 0.12
ZONE_OVERLAY_BASE = 0.88
IN_ZONE_BOX_COLOR = (0, 255, 0)
TRAJECTORY_OUTLINE_COLOR = (40, 40, 40)
LINGERING_ORANGE = (0, 165, 255)
LINGERING_RED = (0, 0, 255)
HUD_GREEN = (0, 255, 0)
HUD_WHITE = (255, 255, 255)
HUD_RED = (0, 0, 255)


def get_class_color(class_id: int) -> tuple[int, int, int]:
    name = COCO_CLASS_NAMES[class_id] if class_id < len(COCO_CLASS_NAMES) else "unknown"
    if name == "person":
        return CLASS_COLORS["person"]
    if name in VEHICLE_CLASS_NAMES:
        return CLASS_COLORS["vehicle"]
    if name in ANIMAL_CLASS_NAMES:
        return CLASS_COLORS["animal"]
    return CLASS_COLORS["other"]
