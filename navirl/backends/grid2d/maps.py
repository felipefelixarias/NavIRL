from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from navirl.backends.grid2d.constants import FREE_SPACE, OBSTACLE_SPACE


def hallway_map() -> np.ndarray:
    m = np.zeros((220, 360), dtype=np.uint8)
    m[60:160, 20:340] = FREE_SPACE
    return m


def doorway_map() -> np.ndarray:
    m = np.zeros((240, 320), dtype=np.uint8)
    m[30:210, 20:130] = FREE_SPACE
    m[30:210, 190:300] = FREE_SPACE
    m[100:140, 130:190] = FREE_SPACE
    return m


def kitchen_map() -> np.ndarray:
    m = np.zeros((300, 340), dtype=np.uint8)
    m[20:280, 20:320] = FREE_SPACE
    # counters / islands
    m[70:110, 70:260] = OBSTACLE_SPACE
    m[180:220, 80:230] = OBSTACLE_SPACE
    m[110:180, 260:300] = OBSTACLE_SPACE
    return m


def group_map() -> np.ndarray:
    m = np.zeros((260, 360), dtype=np.uint8)
    m[20:240, 20:340] = FREE_SPACE
    m[110:150, 165:195] = OBSTACLE_SPACE
    return m


def comfort_map() -> np.ndarray:
    m = np.zeros((260, 360), dtype=np.uint8)
    m[70:190, 20:340] = FREE_SPACE
    m[70:100, 140:220] = OBSTACLE_SPACE
    m[160:190, 140:220] = OBSTACLE_SPACE
    return m


def apartment_micro_map() -> np.ndarray:
    m = np.zeros((300, 360), dtype=np.uint8)
    # bedroom
    m[20:130, 20:140] = FREE_SPACE
    # hall
    m[70:110, 140:240] = FREE_SPACE
    # kitchen
    m[20:170, 240:340] = FREE_SPACE
    # dining/living
    m[170:280, 140:340] = FREE_SPACE
    # connector
    m[130:200, 100:180] = FREE_SPACE
    # small furniture obstacles
    m[45:75, 40:85] = OBSTACLE_SPACE
    m[200:235, 215:285] = OBSTACLE_SPACE
    return m


def hospital_corridor_map() -> np.ndarray:
    m = np.zeros((280, 400), dtype=np.uint8)
    # main corridor
    m[50:230, 30:370] = FREE_SPACE
    # patient rooms (alternating on both sides)
    m[20:50, 50:120] = FREE_SPACE  # room 1
    m[20:50, 160:230] = FREE_SPACE  # room 2
    m[20:50, 270:340] = FREE_SPACE  # room 3
    m[230:260, 70:140] = FREE_SPACE  # room 4
    m[230:260, 180:250] = FREE_SPACE  # room 5
    m[230:260, 290:360] = FREE_SPACE  # room 6
    # nursing station (central)
    m[110:170, 150:250] = FREE_SPACE
    # medical equipment obstacles
    m[130:150, 100:130] = OBSTACLE_SPACE  # equipment cart
    m[130:150, 270:300] = OBSTACLE_SPACE  # another cart
    return m


DEFAULT_PIXELS_PER_METER = 100.0


@dataclass(slots=True)
class MapInfo:
    binary_map: np.ndarray
    source: str
    map_id: str
    map_path: str | None
    pixels_per_meter: float
    meters_per_pixel: float
    width_px: int
    height_px: int
    width_m: float
    height_m: float
    scale_explicit: bool
    downsample: float = 1.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "id": self.map_id,
            "path": self.map_path,
            "pixels_per_meter": float(self.pixels_per_meter),
            "meters_per_pixel": float(self.meters_per_pixel),
            "width_px": int(self.width_px),
            "height_px": int(self.height_px),
            "width_m": float(self.width_m),
            "height_m": float(self.height_m),
            "scale_explicit": bool(self.scale_explicit),
            "downsample": float(self.downsample),
            "world_units": "meters",
        }


BUILTIN_MAPS = {
    "hallway": {"factory": hallway_map, "pixels_per_meter": 100.0},
    "doorway": {"factory": doorway_map, "pixels_per_meter": 100.0},
    "kitchen": {"factory": kitchen_map, "pixels_per_meter": 100.0},
    "group": {"factory": group_map, "pixels_per_meter": 100.0},
    "comfort": {"factory": comfort_map, "pixels_per_meter": 100.0},
    "apartment_micro": {"factory": apartment_micro_map, "pixels_per_meter": 100.0},
    "hospital_corridor": {"factory": hospital_corridor_map, "pixels_per_meter": 100.0},
}


def _resolve_map_path(path_str: str, base_dir: Path | None) -> Path:
    path = Path(path_str)
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    return path


def _resolve_scale(
    map_cfg: dict,
    default_pixels_per_meter: float,
    *,
    require_explicit: bool,
) -> tuple[float, float, bool]:
    ppm_raw = map_cfg.get("pixels_per_meter")
    mpp_raw = map_cfg.get("meters_per_pixel")

    if ppm_raw is None and mpp_raw is None:
        if require_explicit:
            raise ValueError(
                "scene.map.pixels_per_meter or scene.map.meters_per_pixel is required "
                "for source=path"
            )
        ppm = float(default_pixels_per_meter)
        explicit = False
    elif ppm_raw is not None and mpp_raw is not None:
        ppm = float(ppm_raw)
        mpp = float(mpp_raw)
        if ppm <= 0.0 or mpp <= 0.0:
            raise ValueError("Map scale must be positive.")
        derived_mpp = 1.0 / ppm
        if abs(mpp - derived_mpp) > max(1e-9, derived_mpp * 0.02):
            raise ValueError(
                f"Inconsistent map scale: pixels_per_meter={ppm} and meters_per_pixel={mpp}."
            )
        explicit = True
    elif ppm_raw is not None:
        ppm = float(ppm_raw)
        if ppm <= 0.0:
            raise ValueError("scene.map.pixels_per_meter must be > 0")
        explicit = True
    else:
        mpp = float(mpp_raw)
        if mpp <= 0.0:
            raise ValueError("scene.map.meters_per_pixel must be > 0")
        ppm = 1.0 / mpp
        explicit = True

    if ppm <= 0.0:
        raise ValueError("Map pixels_per_meter must be > 0")

    return float(ppm), float(1.0 / ppm), explicit


def _apply_downsample(binary: np.ndarray, downsample: float) -> np.ndarray:
    if downsample <= 1.0:
        return binary.astype(np.uint8)

    h, w = binary.shape[:2]
    out_w = max(1, int(round(w / downsample)))
    out_h = max(1, int(round(h / downsample)))
    if out_w == w and out_h == h:
        return binary.astype(np.uint8)
    return cv2.resize(binary.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def load_map_info(scene_cfg: dict, base_dir: Path | None = None) -> MapInfo:
    map_cfg = dict(scene_cfg.get("map", {}))
    source = str(map_cfg.get("source", "builtin"))

    if source == "builtin":
        map_id = str(map_cfg.get("id", "hallway"))
        if map_id not in BUILTIN_MAPS:
            raise ValueError(f"Unknown builtin map '{map_id}'. Available: {sorted(BUILTIN_MAPS)}")
        entry = BUILTIN_MAPS[map_id]
        ppm, mpp, explicit = _resolve_scale(
            map_cfg,
            default_pixels_per_meter=float(entry["pixels_per_meter"]),
            require_explicit=False,
        )
        binary = entry["factory"]().astype(np.uint8)
        downsample = float(map_cfg.get("downsample", 1.0))
        if downsample <= 0.0:
            raise ValueError("scene.map.downsample must be > 0")
        binary = _apply_downsample(binary, downsample)
        ppm = ppm / downsample
        mpp = 1.0 / ppm
        h, w = binary.shape[:2]
        return MapInfo(
            binary_map=binary,
            source="builtin",
            map_id=map_id,
            map_path=None,
            pixels_per_meter=ppm,
            meters_per_pixel=mpp,
            width_px=int(w),
            height_px=int(h),
            width_m=float(w * mpp),
            height_m=float(h * mpp),
            scale_explicit=explicit,
            downsample=float(downsample),
        )

    if source == "path":
        path_str = map_cfg.get("path")
        if not path_str:
            raise ValueError("scene.map.path is required when source=path")
        path = _resolve_map_path(str(path_str), base_dir)
        map_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if map_gray is None:
            raise FileNotFoundError(f"Unable to read map image: {path}")
        ppm, mpp, explicit = _resolve_scale(
            map_cfg,
            default_pixels_per_meter=DEFAULT_PIXELS_PER_METER,
            require_explicit=True,
        )
        binary = (map_gray > 127).astype(np.uint8)
        downsample = float(map_cfg.get("downsample", 1.0))
        if downsample <= 0.0:
            raise ValueError("scene.map.downsample must be > 0")
        binary = _apply_downsample(binary, downsample)
        ppm = ppm / downsample
        mpp = 1.0 / ppm
        h, w = binary.shape[:2]
        return MapInfo(
            binary_map=binary,
            source="path",
            map_id=str(path.stem),
            map_path=str(path),
            pixels_per_meter=ppm,
            meters_per_pixel=mpp,
            width_px=int(w),
            height_px=int(h),
            width_m=float(w * mpp),
            height_m=float(h * mpp),
            scale_explicit=explicit,
            downsample=float(downsample),
        )

    raise ValueError(f"Unsupported scene.map.source '{source}'")


def load_map(scene_cfg: dict, base_dir: Path | None = None) -> np.ndarray:
    return load_map_info(scene_cfg, base_dir=base_dir).binary_map
