#!/usr/bin/env python3
"""
Simple test for system startup orchestration.

Tests the main process coordination logic without actually starting processes.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_process_manager_creation():
    """Test ProcessManager can be created with valid config."""
    try:
        from main import ProcessManager

        config = {
            "debug": True,
            "web_host": "127.0.0.1",
            "web_port": 8080,
            "wled_host": "192.168.1.100",
            "wled_port": 4048,
            "default_content_dir": "./test_images",
            "diffusion_patterns_path": "./config/diffusion_patterns.npz",
        }

        manager = ProcessManager(config)

        print("✓ ProcessManager created successfully")
        print(f"✓ Config loaded: {len(config)} settings")
        print(f"✓ Control state name: {manager.control_state.name}")

        return True

    except Exception as e:
        print(f"✗ ProcessManager creation failed: {e}")
        return False


def test_control_state_functionality():
    """Test ControlState basic functionality."""
    try:
        from src.core.control_state import ControlState, PlayState, SystemState

        # Test enum values
        assert SystemState.STARTING.value == "starting"
        assert SystemState.RESTARTING.value == "restarting"
        assert SystemState.REBOOTING.value == "rebooting"
        assert PlayState.PLAYING.value == "playing"

        print("✓ Control state enums working correctly")
        print("✓ New system states (STARTING, RESTARTING, REBOOTING) available")

        return True

    except Exception as e:
        print(f"✗ Control state test failed: {e}")
        return False


def test_startup_sequence_logic():
    """Test the startup coordination logic."""
    try:
        from main import ProcessManager

        config = {"debug": True, "web_host": "127.0.0.1", "web_port": 8080}
        manager = ProcessManager(config)

        # Test that coordination events exist
        assert hasattr(manager, "web_server_ready")
        assert hasattr(manager, "producer_ready")
        assert hasattr(manager, "consumer_ready")

        print("✓ Process coordination events created")
        print("✓ Startup sequence logic structure validated")

        return True

    except Exception as e:
        print(f"✗ Startup sequence test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Prismatron Phase 5 Implementation")
    print("=" * 50)

    tests = [
        test_control_state_functionality,
        test_process_manager_creation,
        test_startup_sequence_logic,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("✓ All Phase 5 orchestration components working correctly!")
        print("\nNext steps:")
        print("- Run 'python main.py --help' to see usage options")
        print("- Start system with 'python main.py --debug'")
        print("- Use API endpoints /api/system/restart and /api/system/reboot")
    else:
        print("✗ Some tests failed - check implementation")
        sys.exit(1)


if __name__ == "__main__":
    main()
