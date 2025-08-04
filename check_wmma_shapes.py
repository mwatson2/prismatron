#!/usr/bin/env python3
"""
Check which WMMA shapes are supported on the current GPU.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    import cupy
    print(f"CUDA available: {cupy.cuda.runtime.runtimeGetVersion()}")

    # Get GPU info
    device = cupy.cuda.Device()
    capability = device.compute_capability
    print(f"GPU Compute Capability: {capability[0]}.{capability[1]}")

    if int(capability[0]) >= 8:
        print("✅ GPU supports WMMA operations (Ampere or newer)")

        # List supported WMMA shapes for half precision
        print("\nSupported WMMA shapes for half precision:")
        print("Standard shapes:")
        print("- 16x16x16 (widely supported)")
        print("- 32x8x16 (supported on sm_80+)")
        print("- 8x32x16 (supported on sm_80+)")

        if (int(capability[0]), int(capability[1])) >= (8, 0):
            print("✅ Your GPU supports 8x32x16 and 32x8x16 shapes")
            print("✅ Can implement true 8x32x8 operations using 8x32x16 shapes")
        else:
            print("⚠️  Your GPU may only support 16x16x16 shapes")

        print("\nNote: 8x32x8 specifically requires using 8x32x16 with K=8 padding")

    else:
        print("❌ GPU does not support WMMA operations (requires sm_70+)")

except ImportError:
    print("❌ CuPy not available")
    sys.exit(1)
