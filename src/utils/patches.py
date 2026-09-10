"""PyTorch + Ultralytics safe-globals patch for torch>=2.6 checkpoints."""

import importlib

import torch


def _resolve_obj(dotted_path: str):
    """Resolve `pkg.mod.Symbol` into the actual Python object."""
    module_name, _, attr_name = dotted_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid dotted path: {dotted_path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _register_safe_globals(paths):
    """Register resolvable objects to torch safe globals and return count."""
    add_safe = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe is None:
        print("[patch] torch.serialization.add_safe_globals unavailable, skip")
        return 0

    registered = 0
    for path in paths:
        try:
            obj = _resolve_obj(path)
            add_safe([obj])
            registered += 1
        except Exception as exc:
            # Best-effort patching: skip missing symbols from version differences.
            print(f"[patch] skip {path}: {exc}")
    return registered


def _patch_torch_safe_globals():
    """Register core torch/python objects often referenced by model pickles."""
    core_paths = [
        "torch._utils._rebuild_tensor_v2",
        "torch.nn.modules.container.ModuleList",
        "torch.nn.modules.conv.Conv2d",
        "collections.OrderedDict",
    ]
    return _register_safe_globals(core_paths)


def _patch_yolo_safe_globals():
    """Register ultralytics model/task blocks used in YOLO checkpoints."""
    yolo_paths = [
        "ultralytics.nn.tasks.DetectionModel",
        "ultralytics.nn.tasks.SegmentationModel",
        "ultralytics.nn.tasks.PoseModel",
        "ultralytics.nn.modules.Conv",
        "ultralytics.nn.modules.Bottleneck",
        "ultralytics.nn.modules.C2f",
        "ultralytics.nn.modules.SPPF",
    ]
    return _register_safe_globals(yolo_paths)


def apply_all():
    """Apply all load patches once at startup."""
    try:
        count_core = _patch_torch_safe_globals()
        count_yolo = _patch_yolo_safe_globals()
        print(f"[patch] applied safe globals: core={count_core}, yolo={count_yolo}")
        return True
    except Exception as exc:
        print(f"[patch] failed: {exc}")
        return False


if __name__ == "__main__":
    apply_all()
