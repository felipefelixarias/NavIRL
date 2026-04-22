"""Tests for navirl.backends.grid2d.maps.

Covers the seven builtin map factories, MapInfo.to_dict(), the internal
helpers _resolve_map_path / _resolve_scale / _apply_downsample, and the
public load_map_info / load_map functions across builtin and path sources.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from navirl.backends.grid2d.constants import FREE_SPACE, OBSTACLE_SPACE
from navirl.backends.grid2d.maps import (
    BUILTIN_MAPS,
    DEFAULT_PIXELS_PER_METER,
    MapInfo,
    _apply_downsample,
    _resolve_map_path,
    _resolve_scale,
    apartment_micro_map,
    comfort_map,
    doorway_map,
    group_map,
    hallway_map,
    hospital_corridor_map,
    kitchen_map,
    load_map,
    load_map_info,
)

# ---------------------------------------------------------------------------
# Builtin map factories
# ---------------------------------------------------------------------------


class TestBuiltinMapFactories:
    """Each builtin factory returns a uint8 occupancy grid with both free and
    obstacle pixels and the documented overall shape."""

    @pytest.mark.parametrize(
        "factory, expected_shape",
        [
            (hallway_map, (220, 360)),
            (doorway_map, (240, 320)),
            (kitchen_map, (300, 340)),
            (group_map, (260, 360)),
            (comfort_map, (260, 360)),
            (apartment_micro_map, (300, 360)),
            (hospital_corridor_map, (280, 400)),
        ],
    )
    def test_shape_and_dtype(self, factory, expected_shape):
        m = factory()
        assert m.shape == expected_shape
        assert m.dtype == np.uint8

    @pytest.mark.parametrize(
        "factory",
        [hallway_map, doorway_map, kitchen_map, group_map, comfort_map,
         apartment_micro_map, hospital_corridor_map],
    )
    def test_has_free_pixels(self, factory):
        m = factory()
        assert (m == FREE_SPACE).any()

    @pytest.mark.parametrize(
        "factory",
        [doorway_map, kitchen_map, group_map, comfort_map,
         apartment_micro_map, hospital_corridor_map],
    )
    def test_has_obstacle_pixels(self, factory):
        # All these maps have explicit walls or interior obstacles
        m = factory()
        assert (m == OBSTACLE_SPACE).any()

    def test_hallway_is_pure_corridor(self):
        # hallway_map() draws a free rectangle on a fully-obstacle background
        m = hallway_map()
        assert (m == OBSTACLE_SPACE).any()
        assert m[100, 100] == FREE_SPACE  # interior of corridor
        assert m[0, 0] == OBSTACLE_SPACE  # corner outside corridor

    def test_doorway_has_narrow_passage(self):
        m = doorway_map()
        # The passage band is at rows 100-140 between the two rooms
        assert m[120, 160] == FREE_SPACE
        # Above and below the passage in the wall region is obstacle
        assert m[50, 160] == OBSTACLE_SPACE


class TestBuiltinMapsRegistry:
    def test_all_seven_registered(self):
        expected = {
            "hallway", "doorway", "kitchen", "group", "comfort",
            "apartment_micro", "hospital_corridor",
        }
        assert set(BUILTIN_MAPS) == expected

    def test_each_entry_has_factory_and_ppm(self):
        for name, entry in BUILTIN_MAPS.items():
            assert callable(entry["factory"]), name
            assert entry["pixels_per_meter"] > 0, name


# ---------------------------------------------------------------------------
# MapInfo dataclass
# ---------------------------------------------------------------------------


def _make_mapinfo(**overrides):
    defaults = dict(
        binary_map=np.ones((4, 4), dtype=np.uint8),
        source="builtin",
        map_id="test",
        map_path=None,
        pixels_per_meter=100.0,
        meters_per_pixel=0.01,
        width_px=4,
        height_px=4,
        width_m=0.04,
        height_m=0.04,
        scale_explicit=False,
        downsample=1.0,
    )
    defaults.update(overrides)
    return MapInfo(**defaults)


class TestMapInfoToDict:
    def test_round_trip_keys(self):
        info = _make_mapinfo()
        d = info.to_dict()
        for key in ("source", "id", "path", "pixels_per_meter",
                    "meters_per_pixel", "width_px", "height_px",
                    "width_m", "height_m", "scale_explicit",
                    "downsample", "world_units"):
            assert key in d

    def test_world_units_is_meters(self):
        assert _make_mapinfo().to_dict()["world_units"] == "meters"

    def test_id_field_uses_map_id(self):
        d = _make_mapinfo(map_id="hallway").to_dict()
        assert d["id"] == "hallway"

    def test_numeric_fields_are_python_floats_or_ints(self):
        d = _make_mapinfo().to_dict()
        assert isinstance(d["pixels_per_meter"], float)
        assert isinstance(d["meters_per_pixel"], float)
        assert isinstance(d["width_px"], int)
        assert isinstance(d["height_px"], int)
        assert isinstance(d["downsample"], float)

    def test_path_passed_through(self):
        d = _make_mapinfo(map_path="/tmp/foo.png").to_dict()
        assert d["path"] == "/tmp/foo.png"


# ---------------------------------------------------------------------------
# _resolve_map_path
# ---------------------------------------------------------------------------


class TestResolveMapPath:
    def test_absolute_path_unchanged(self, tmp_path):
        abs_path = tmp_path / "x.png"
        result = _resolve_map_path(str(abs_path), base_dir=Path("/var/tmp"))
        assert result == abs_path

    def test_relative_path_resolved_against_base_dir(self, tmp_path):
        result = _resolve_map_path("subdir/x.png", base_dir=tmp_path)
        # resolve() makes it absolute; just check the base is included
        assert result.is_absolute()
        assert "subdir" in result.parts
        assert result.name == "x.png"

    def test_relative_path_no_base_dir_returns_relative(self):
        # When base_dir is None, the path is returned as-is (still relative)
        result = _resolve_map_path("relative/x.png", base_dir=None)
        assert result == Path("relative/x.png")


# ---------------------------------------------------------------------------
# _resolve_scale
# ---------------------------------------------------------------------------


class TestResolveScale:
    def test_neither_with_default(self):
        ppm, mpp, explicit = _resolve_scale(
            {}, default_pixels_per_meter=50.0, require_explicit=False
        )
        assert ppm == 50.0
        assert mpp == pytest.approx(0.02)
        assert explicit is False

    def test_neither_with_require_explicit_raises(self):
        with pytest.raises(ValueError, match="pixels_per_meter or"):
            _resolve_scale({}, default_pixels_per_meter=50.0, require_explicit=True)

    def test_only_pixels_per_meter(self):
        ppm, mpp, explicit = _resolve_scale(
            {"pixels_per_meter": 200.0}, default_pixels_per_meter=1.0, require_explicit=False
        )
        assert ppm == 200.0
        assert mpp == pytest.approx(1.0 / 200.0)
        assert explicit is True

    def test_only_pixels_per_meter_zero_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            _resolve_scale(
                {"pixels_per_meter": 0.0},
                default_pixels_per_meter=1.0,
                require_explicit=False,
            )

    def test_only_pixels_per_meter_negative_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            _resolve_scale(
                {"pixels_per_meter": -10.0},
                default_pixels_per_meter=1.0,
                require_explicit=False,
            )

    def test_only_meters_per_pixel(self):
        ppm, mpp, explicit = _resolve_scale(
            {"meters_per_pixel": 0.05},
            default_pixels_per_meter=1.0,
            require_explicit=False,
        )
        assert ppm == pytest.approx(20.0)
        assert mpp == pytest.approx(0.05)
        assert explicit is True

    def test_only_meters_per_pixel_zero_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            _resolve_scale(
                {"meters_per_pixel": 0.0},
                default_pixels_per_meter=1.0,
                require_explicit=False,
            )

    def test_only_meters_per_pixel_negative_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            _resolve_scale(
                {"meters_per_pixel": -0.1},
                default_pixels_per_meter=1.0,
                require_explicit=False,
            )

    def test_both_consistent(self):
        ppm, mpp, explicit = _resolve_scale(
            {"pixels_per_meter": 100.0, "meters_per_pixel": 0.01},
            default_pixels_per_meter=1.0,
            require_explicit=False,
        )
        assert ppm == 100.0
        assert mpp == pytest.approx(0.01)
        assert explicit is True

    def test_both_inconsistent_raises(self):
        with pytest.raises(ValueError, match="Inconsistent map scale"):
            _resolve_scale(
                {"pixels_per_meter": 100.0, "meters_per_pixel": 0.05},  # 1/100 != 0.05
                default_pixels_per_meter=1.0,
                require_explicit=False,
            )

    def test_both_within_two_percent_tolerance(self):
        # 1.5% off — within tolerance
        ppm, mpp, explicit = _resolve_scale(
            {"pixels_per_meter": 100.0, "meters_per_pixel": 0.0101},
            default_pixels_per_meter=1.0,
            require_explicit=False,
        )
        assert ppm == 100.0
        assert explicit is True

    def test_both_zero_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            _resolve_scale(
                {"pixels_per_meter": 0.0, "meters_per_pixel": 0.0},
                default_pixels_per_meter=1.0,
                require_explicit=False,
            )


# ---------------------------------------------------------------------------
# _apply_downsample
# ---------------------------------------------------------------------------


class TestApplyDownsample:
    def test_no_op_when_factor_is_one(self):
        binary = (np.random.default_rng(0).integers(0, 2, size=(20, 30))).astype(np.uint8)
        out = _apply_downsample(binary, 1.0)
        assert out.shape == binary.shape
        assert np.array_equal(out, binary)

    def test_no_op_when_factor_below_one(self):
        binary = np.ones((10, 10), dtype=np.uint8)
        out = _apply_downsample(binary, 0.5)
        assert out.shape == (10, 10)

    def test_returns_uint8_when_input_is_other_dtype(self):
        binary = np.ones((10, 10), dtype=np.uint16)
        out = _apply_downsample(binary, 1.0)
        assert out.dtype == np.uint8

    def test_factor_two_halves_dimensions(self):
        binary = np.ones((20, 40), dtype=np.uint8)
        out = _apply_downsample(binary, 2.0)
        assert out.shape == (10, 20)
        assert out.dtype == np.uint8

    def test_factor_results_in_at_least_one_pixel(self):
        binary = np.ones((4, 4), dtype=np.uint8)
        out = _apply_downsample(binary, 100.0)
        assert out.shape[0] >= 1 and out.shape[1] >= 1

    def test_no_op_when_rounded_size_matches(self):
        # downsample 1.4 on a 1x1 -> rounds to 1x1 -> short-circuit branch
        binary = np.ones((1, 1), dtype=np.uint8)
        out = _apply_downsample(binary, 1.4)
        assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# load_map_info: builtin source
# ---------------------------------------------------------------------------


class TestLoadMapInfoBuiltin:
    @pytest.mark.parametrize("name", sorted(BUILTIN_MAPS))
    def test_each_builtin_loads(self, name):
        info = load_map_info({"map": {"source": "builtin", "id": name}})
        assert info.source == "builtin"
        assert info.map_id == name
        assert info.map_path is None
        assert info.binary_map.dtype == np.uint8
        assert info.pixels_per_meter > 0
        assert info.meters_per_pixel == pytest.approx(1.0 / info.pixels_per_meter)
        h, w = info.binary_map.shape
        assert info.width_px == w
        assert info.height_px == h
        assert info.width_m == pytest.approx(w / info.pixels_per_meter)
        assert info.height_m == pytest.approx(h / info.pixels_per_meter)
        assert info.downsample == 1.0
        assert info.scale_explicit is False

    def test_default_source_is_builtin_hallway(self):
        # Both source and id default
        info = load_map_info({"map": {}})
        assert info.source == "builtin"
        assert info.map_id == "hallway"

    def test_no_map_key_uses_defaults(self):
        # Empty scene_cfg also defaults to builtin/hallway
        info = load_map_info({})
        assert info.source == "builtin"
        assert info.map_id == "hallway"

    def test_unknown_builtin_raises(self):
        with pytest.raises(ValueError, match="Unknown builtin map"):
            load_map_info({"map": {"source": "builtin", "id": "no_such_map"}})

    def test_explicit_scale_overrides_default(self):
        info = load_map_info(
            {"map": {"source": "builtin", "id": "hallway", "pixels_per_meter": 50.0}}
        )
        assert info.pixels_per_meter == 50.0
        assert info.scale_explicit is True

    def test_downsample_halves_resolution(self):
        info_full = load_map_info({"map": {"source": "builtin", "id": "hallway"}})
        info_ds = load_map_info(
            {"map": {"source": "builtin", "id": "hallway", "downsample": 2.0}}
        )
        assert info_ds.downsample == 2.0
        assert info_ds.pixels_per_meter == pytest.approx(info_full.pixels_per_meter / 2.0)
        # World extents should be preserved (within rounding)
        assert info_ds.width_m == pytest.approx(info_full.width_m, rel=0.02)
        assert info_ds.height_m == pytest.approx(info_full.height_m, rel=0.02)

    def test_downsample_zero_raises(self):
        with pytest.raises(ValueError, match="downsample must be > 0"):
            load_map_info(
                {"map": {"source": "builtin", "id": "hallway", "downsample": 0.0}}
            )

    def test_downsample_negative_raises(self):
        with pytest.raises(ValueError, match="downsample must be > 0"):
            load_map_info(
                {"map": {"source": "builtin", "id": "hallway", "downsample": -1.0}}
            )


# ---------------------------------------------------------------------------
# load_map_info: path source
# ---------------------------------------------------------------------------


@pytest.fixture()
def png_map(tmp_path):
    """Write a 40x60 grayscale PNG with mixed free/obstacle pixels."""
    img = np.zeros((40, 60), dtype=np.uint8)
    img[5:35, 5:55] = 255  # bright = free
    p = tmp_path / "tiny.png"
    cv2.imwrite(str(p), img)
    return p


class TestLoadMapInfoPath:
    def test_loads_existing_png(self, png_map):
        info = load_map_info(
            {"map": {
                "source": "path",
                "path": str(png_map),
                "pixels_per_meter": 80.0,
            }}
        )
        assert info.source == "path"
        assert info.map_id == png_map.stem
        assert info.map_path == str(png_map)
        assert info.pixels_per_meter == 80.0
        assert info.scale_explicit is True
        assert info.binary_map.shape == (40, 60)
        # Pixels above 127 should be free (1), others obstacle (0)
        assert (info.binary_map == FREE_SPACE).any()
        assert (info.binary_map == OBSTACLE_SPACE).any()

    def test_relative_path_resolved_via_base_dir(self, png_map):
        info = load_map_info(
            {"map": {
                "source": "path",
                "path": png_map.name,
                "pixels_per_meter": 80.0,
            }},
            base_dir=png_map.parent,
        )
        assert Path(info.map_path).resolve() == png_map.resolve()

    def test_meters_per_pixel_input(self, png_map):
        info = load_map_info(
            {"map": {
                "source": "path",
                "path": str(png_map),
                "meters_per_pixel": 0.02,
            }}
        )
        assert info.pixels_per_meter == pytest.approx(50.0)

    def test_missing_path_raises(self):
        with pytest.raises(ValueError, match="scene.map.path is required"):
            load_map_info({"map": {"source": "path", "pixels_per_meter": 100.0}})

    def test_unreadable_path_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Unable to read map image"):
            load_map_info({"map": {
                "source": "path",
                "path": str(tmp_path / "does_not_exist.png"),
                "pixels_per_meter": 100.0,
            }})

    def test_missing_scale_raises(self, png_map):
        with pytest.raises(ValueError, match="scene.map.pixels_per_meter or"):
            load_map_info({"map": {"source": "path", "path": str(png_map)}})

    def test_downsample_zero_raises(self, png_map):
        with pytest.raises(ValueError, match="downsample must be > 0"):
            load_map_info({"map": {
                "source": "path",
                "path": str(png_map),
                "pixels_per_meter": 100.0,
                "downsample": 0.0,
            }})

    def test_downsample_halves_resolution(self, png_map):
        info_full = load_map_info({"map": {
            "source": "path",
            "path": str(png_map),
            "pixels_per_meter": 80.0,
        }})
        info_ds = load_map_info({"map": {
            "source": "path",
            "path": str(png_map),
            "pixels_per_meter": 80.0,
            "downsample": 2.0,
        }})
        assert info_ds.pixels_per_meter == pytest.approx(40.0)
        assert info_ds.binary_map.shape[0] <= info_full.binary_map.shape[0]


# ---------------------------------------------------------------------------
# load_map_info: invalid source
# ---------------------------------------------------------------------------


class TestLoadMapInfoInvalidSource:
    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unsupported scene.map.source"):
            load_map_info({"map": {"source": "definitely_not_a_source"}})


# ---------------------------------------------------------------------------
# load_map convenience wrapper
# ---------------------------------------------------------------------------


class TestLoadMap:
    def test_returns_binary_map_array(self):
        m = load_map({"map": {"source": "builtin", "id": "hallway"}})
        assert isinstance(m, np.ndarray)
        assert m.dtype == np.uint8

    def test_matches_load_map_info(self):
        cfg = {"map": {"source": "builtin", "id": "kitchen"}}
        m = load_map(cfg)
        info = load_map_info(cfg)
        assert np.array_equal(m, info.binary_map)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_default_pixels_per_meter_value():
    # Sanity check on the default exposed by the module
    assert DEFAULT_PIXELS_PER_METER == 100.0
