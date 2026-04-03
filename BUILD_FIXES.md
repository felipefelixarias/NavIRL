# Build Fixes Applied

This document records the fixes applied to resolve NavIRL build issues.

## Issues Resolved

1. **Missing cmake**: The project requires cmake >= 3.16 to build the RVO2 C++ library
2. **Missing OpenCV system dependencies**: opencv-python requires system graphics libraries
3. **Externally managed Python environment**: Ubuntu 24.04 requires virtual environments

## Fixes Applied

### 1. Install cmake
```bash
sudo apt update && sudo apt install -y cmake
```

### 2. Install OpenCV System Dependencies
```bash
sudo apt install -y libgl1-mesa-dev libglib2.0-0
```

### 3. Create Virtual Environment and Install Project
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

## System Requirements

- Ubuntu 24.04+ or similar
- Python 3.12
- cmake >= 3.16
- System graphics libraries for OpenCV
- C++ compiler (g++)

The project now builds successfully with these dependencies installed.
