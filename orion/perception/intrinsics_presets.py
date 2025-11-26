"""Device-specific camera intrinsics presets."""

from typing import Dict

from .types import CameraIntrinsics


# Presets derived from known demo configurations.
INTRINSICS_PRESETS: Dict[str, CameraIntrinsics] = {
    "demo_room_iphone15pro_main": CameraIntrinsics(
        fx=1215.0,
        fy=1110.0,
        cx=960.0,
        cy=540.0,
        width=1920,
        height=1080,
    ),
    "legacy_placeholder_640x480": CameraIntrinsics(
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        width=640,
        height=480,
    ),
}


DEFAULT_INTRINSICS_PRESET = "demo_room_iphone15pro_main"
