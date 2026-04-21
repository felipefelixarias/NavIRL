# Build Fixes Applied

This document records the fixes applied to resolve NavIRL build issues.

## Issues Resolved

1. **Missing cmake**: The project requires cmake >= 3.16 to build the RVO2 C++ library
2. **Missing CPython headers**: the rvo2 Cython extension fails with `Python.h: No such file or directory` when `python3-dev` is absent
3. **Missing OpenCV system dependencies**: opencv-python requires system graphics libraries (e.g. `libxcb.so.1`); on headless containers this surfaces as ~30 import-error collection failures across unrelated test modules because `navirl.backends.grid2d.backend` imports `cv2` at module load
4. **Externally managed Python environment**: Debian/Ubuntu's PEP 668 marker requires virtual environments

## Fixes Applied

### 1. Install cmake
```bash
sudo apt update && sudo apt install -y cmake
```

If you cannot use `apt`, install cmake into the venv instead:
```bash
.venv/bin/python -m pip install cmake   # binary at .venv/bin/cmake
```

### 2. Install Python development headers
Required to build the rvo2 Cython extension (`pip install -e .` invokes the C++ build).
```bash
sudo apt install -y python3-dev
```

### 3. Install OpenCV runtime
Pick one of:

**Option A — system graphics libraries (matches `opencv-python` wheel):**
```bash
sudo apt install -y libgl1-mesa-dev libglib2.0-0 libxcb1
```

**Option B — switch to the headless wheel (recommended for CI / containers without a display):**
```bash
python -m pip uninstall -y opencv-python
python -m pip install opencv-python-headless
```
Both wheels expose the same `cv2` API; the headless wheel drops the GUI/X11 deps that NavIRL does not use.

### 4. Create Virtual Environment and Install Project
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[dev]
```

## Verification

The following tests pass successfully:
- `python -m navirl --help` - CLI works
- `python -m pytest tests/test_smoke.py -v` - Basic smoke test
- `python -m pytest tests/test_scenarios.py -v` - Scenario validation
- `python -m pytest` - Full suite (~5388 passing, ~125 skipped for optional deps like PyTorch/gymnasium)

## System Requirements

- Ubuntu 24.04+ / Debian 12+ or similar
- Python 3.11 or 3.12
- cmake >= 3.16
- `python3-dev` (CPython headers)
- OpenCV runtime — either system X11/GL libs OR the `opencv-python-headless` wheel
- C++ compiler (g++)

The project builds successfully with these dependencies installed.
