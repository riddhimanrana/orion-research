"""Compatibility shim for older YOLO detector import path.

Old tests/scripts import `orion.perception.detectors.yolo_detector.YOLODetector`.
The canonical implementation lives in `yolo.py` as `YOLODetector`. This shim
wraps the newer class and provides the older, simpler API used in tests.
"""
from typing import Any, Dict, List, Optional

try:
    from .yolo import YOLODetector as _YOLOImpl
except Exception:  # pragma: no cover - import fallback for test environments
    # If relative import fails, try absolute
    from orion.perception.detectors.yolo import YOLODetector as _YOLOImpl


class YOLODetector:
    """Lightweight wrapper exposing the legacy init/signature.

    Legacy tests pass a `DetectionConfig` object or simple kwargs. We accept
    either and adapt to the underlying implementation.
    """

    def __init__(self, config_or_path: Optional[Any] = None, **kwargs):
        # Support being passed a DetectionConfig dataclass
        model_path = None
        model_name = kwargs.get("model_name", "yolo11m")
        confidence_threshold = kwargs.get("confidence_threshold", 0.25)
        device = kwargs.get("device", "mps")

        # If a config object is provided, try to read its attributes
        if config_or_path is not None and not isinstance(config_or_path, str):
            cfg = config_or_path
            model_path = getattr(cfg, "model_path", None)
            model_name = getattr(cfg, "model", model_name)
            confidence_threshold = getattr(cfg, "confidence_threshold", confidence_threshold)
            device = getattr(cfg, "device", device)
        elif isinstance(config_or_path, str):
            model_path = config_or_path

        # Instantiate real detector
        self._impl = _YOLOImpl(
            model_path=model_path,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        # Mirror some common attributes for backward compatibility
        try:
            self.model_name = getattr(self._impl, "model_name")
        except Exception:
            self.model_name = model_name

    def __getattr__(self, name: str):
        # Delegate attribute access to underlying implementation where possible
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._impl, name)

    def detect(self, frame) -> List[Dict[str, Any]]:
        """Detect objects in a single frame. Returns list of detection dicts."""
        # Underlying API is `detect_frame`; call and return
        return self._impl.detect_frame(frame)

    def detect_video(self, *args, **kwargs):
        return self._impl.detect_video(*args, **kwargs)

    def get_model_info(self):
        return self._impl.get_model_info()


__all__ = ["YOLODetector"]
