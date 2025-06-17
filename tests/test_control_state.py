"""
Unit tests for the Control State Manager.

Tests the lightweight IPC functionality including shared state management,
process coordination, configuration handling, and event synchronization.
"""

import unittest
import multiprocessing as mp
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.control_state import ControlState, PlayState, SystemState, SystemStatus, SHARED_MEMORY_AVAILABLE


@unittest.skipUnless(SHARED_MEMORY_AVAILABLE, "Shared memory not available (requires Python 3.8+)")
class TestControlState(unittest.TestCase):
    """Test cases for ControlState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.control_name = f"test_control_{os.getpid()}_{int(time.time() * 1000000)}"
        self.control_state = ControlState(self.control_name)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'control_state'):
            self.control_state.cleanup()
    
    def test_initialization(self):
        """Test basic initialization of control state."""
        result = self.control_state.initialize()
        self.assertTrue(result, "Control state initialization should succeed")
        
        # Verify we can read status
        status = self.control_state.get_status()
        self.assertIsNotNone(status, "Should be able to read status after initialization")
        self.assertEqual(status.play_state, PlayState.STOPPED)
        self.assertEqual(status.system_state, SystemState.INITIALIZING)
    
    def test_status_dict_format(self):
        """Test that status dictionary has proper format."""
        self.control_state.initialize()
        
        status_dict = self.control_state.get_status_dict()
        
        # Check required fields
        required_fields = [
            'play_state', 'system_state', 'current_file', 'brightness',
            'frame_rate', 'producer_fps', 'consumer_fps', 'error_message',
            'uptime', 'last_update'
        ]
        
        for field in required_fields:
            self.assertIn(field, status_dict, f"Status dict should contain {field}")
        
        # Check enum values are strings
        self.assertIsInstance(status_dict['play_state'], str)
        self.assertIsInstance(status_dict['system_state'], str)
    
    def test_play_state_operations(self):
        """Test play state management."""
        self.control_state.initialize()
        
        # Test setting different play states
        test_states = [PlayState.PLAYING, PlayState.PAUSED, PlayState.STOPPED]
        
        for state in test_states:
            result = self.control_state.set_play_state(state)
            self.assertTrue(result, f"Should successfully set play state to {state}")
            
            status = self.control_state.get_status()
            self.assertEqual(status.play_state, state, f"Play state should be {state}")
    
    def test_current_file_operations(self):
        """Test current file path management."""
        self.control_state.initialize()
        
        test_files = [
            "/path/to/video.mp4",
            "/another/path/image.jpg",
            "relative/path/animation.gif",
            ""  # Empty string
        ]
        
        for filepath in test_files:
            result = self.control_state.set_current_file(filepath)
            self.assertTrue(result, f"Should successfully set current file to {filepath}")
            
            status = self.control_state.get_status()
            self.assertEqual(status.current_file, filepath, f"Current file should be {filepath}")
    
    def test_brightness_operations(self):
        """Test brightness control."""
        self.control_state.initialize()
        
        # Test valid brightness values
        test_values = [0.0, 0.5, 1.0, 0.25, 0.75]
        
        for value in test_values:
            result = self.control_state.set_brightness(value)
            self.assertTrue(result, f"Should successfully set brightness to {value}")
            
            status = self.control_state.get_status()
            self.assertAlmostEqual(status.brightness, value, places=3,
                                 msg=f"Brightness should be {value}")
    
    def test_brightness_clamping(self):
        """Test that brightness values are clamped to valid range."""
        self.control_state.initialize()
        
        # Test out-of-range values
        test_cases = [
            (-1.0, 0.0),   # Below minimum
            (2.0, 1.0),    # Above maximum
            (-0.5, 0.0),   # Negative
            (1.5, 1.0)     # Above 1.0
        ]
        
        for input_value, expected_value in test_cases:
            result = self.control_state.set_brightness(input_value)
            self.assertTrue(result, f"Should successfully clamp brightness {input_value}")
            
            status = self.control_state.get_status()
            self.assertAlmostEqual(status.brightness, expected_value, places=3,
                                 msg=f"Brightness should be clamped to {expected_value}")
    
    def test_frame_rate_monitoring(self):
        """Test frame rate monitoring functionality."""
        self.control_state.initialize()
        
        test_cases = [
            (30.0, 25.0),  # Producer faster than consumer
            (15.0, 20.0),  # Consumer faster than producer
            (30.0, 30.0),  # Same rates
            (0.0, 0.0)     # No activity
        ]
        
        for producer_fps, consumer_fps in test_cases:
            result = self.control_state.set_frame_rates(producer_fps, consumer_fps)
            self.assertTrue(result, f"Should successfully set frame rates")
            
            status = self.control_state.get_status()
            self.assertAlmostEqual(status.producer_fps, producer_fps, places=2)
            self.assertAlmostEqual(status.consumer_fps, consumer_fps, places=2)
            
            # Frame rate should be minimum of producer and consumer
            expected_rate = min(producer_fps, consumer_fps)
            self.assertAlmostEqual(status.frame_rate, expected_rate, places=2)
    
    def test_error_handling(self):
        """Test error state management."""
        self.control_state.initialize()
        
        # Set error state
        error_message = "Test error occurred"
        result = self.control_state.set_error(error_message)
        self.assertTrue(result, "Should successfully set error state")
        
        status = self.control_state.get_status()
        self.assertEqual(status.system_state, SystemState.ERROR)
        self.assertEqual(status.error_message, error_message)
        
        # Clear error state
        result = self.control_state.clear_error()
        self.assertTrue(result, "Should successfully clear error state")
        
        status = self.control_state.get_status()
        self.assertEqual(status.system_state, SystemState.RUNNING)
        self.assertEqual(status.error_message, "")
    
    def test_system_state_transitions(self):
        """Test system state transitions."""
        self.control_state.initialize()
        
        # Test state transitions
        states = [
            SystemState.RUNNING,
            SystemState.SHUTTING_DOWN,
            SystemState.ERROR,
            SystemState.INITIALIZING
        ]
        
        for state in states:
            result = self.control_state.update_system_state(state)
            self.assertTrue(result, f"Should successfully set system state to {state}")
            
            status = self.control_state.get_status()
            self.assertEqual(status.system_state, state)
    
    def test_shutdown_signaling(self):
        """Test shutdown coordination."""
        self.control_state.initialize()
        
        # Initially no shutdown should be requested
        self.assertFalse(self.control_state.is_shutdown_requested())
        
        # Signal shutdown
        self.control_state.signal_shutdown()
        
        # Should now be requested
        self.assertTrue(self.control_state.is_shutdown_requested())
        
        # Status should reflect shutdown state
        status = self.control_state.get_status()
        self.assertEqual(status.system_state, SystemState.SHUTTING_DOWN)
    
    def test_shutdown_wait_timeout(self):
        """Test shutdown wait with timeout."""
        self.control_state.initialize()
        
        # Should timeout when no shutdown signaled
        start_time = time.time()
        result = self.control_state.wait_for_shutdown(timeout=0.1)
        end_time = time.time()
        
        self.assertFalse(result, "Should timeout when no shutdown signaled")
        self.assertGreaterEqual(end_time - start_time, 0.08)  # Should have waited
        self.assertLess(end_time - start_time, 0.3)  # But not too long
    
    def test_timestamp_tracking(self):
        """Test that timestamps are properly maintained."""
        self.control_state.initialize()
        
        start_time = time.time()
        
        # Make a change to trigger timestamp update
        self.control_state.set_play_state(PlayState.PLAYING)
        
        status = self.control_state.get_status()
        
        # Check last_update timestamp
        self.assertGreaterEqual(status.last_update, start_time)
        self.assertLess(status.last_update - start_time, 1.0)
        
        # Check uptime is reasonable
        self.assertGreaterEqual(status.uptime, 0.0)
        self.assertLess(status.uptime, 10.0)  # Should be recent
    
    def test_connection_to_existing_state(self):
        """Test connecting to existing shared state."""
        # Initialize in one instance
        self.control_state.initialize()
        
        # Set some test data
        self.control_state.set_play_state(PlayState.PLAYING)
        self.control_state.set_brightness(0.8)
        self.control_state.set_current_file("/test/file.mp4")
        
        # Create second instance and connect
        consumer_control = ControlState(self.control_name)
        result = consumer_control.connect()
        self.assertTrue(result, "Should successfully connect to existing state")
        
        # Verify data is accessible
        status = consumer_control.get_status()
        self.assertIsNotNone(status)
        self.assertEqual(status.play_state, PlayState.PLAYING)
        self.assertAlmostEqual(status.brightness, 0.8, places=3)
        self.assertEqual(status.current_file, "/test/file.mp4")
        
        # Clean up the consumer (but don't unlink shared memory)
        consumer_control._shared_memory = None
        consumer_control._initialized = False
    
    def test_error_handling_uninitialized(self):
        """Test error handling when not initialized."""
        # Don't initialize
        
        result = self.control_state.set_play_state(PlayState.PLAYING)
        self.assertFalse(result, "Should fail when not initialized")
        
        result = self.control_state.set_brightness(0.5)
        self.assertFalse(result, "Should fail when not initialized")
        
        status = self.control_state.get_status()
        self.assertIsNone(status, "Should return None when not initialized")
    
    def test_cleanup(self):
        """Test proper cleanup of resources."""
        self.control_state.initialize()
        
        # Verify we can read status
        status = self.control_state.get_status()
        self.assertIsNotNone(status)
        
        # Cleanup
        self.control_state.cleanup()
        
        # Should not be able to read status after cleanup
        status = self.control_state.get_status()
        self.assertIsNone(status)


def producer_process(control_name: str, num_updates: int):
    """Producer process for multiprocess testing."""
    try:
        control = ControlState(control_name)
        if not control.connect():
            return False
        
        # Make updates
        for i in range(num_updates):
            control.set_play_state(PlayState.PLAYING if i % 2 == 0 else PlayState.PAUSED)
            control.set_brightness(i / num_updates)
            control.set_current_file(f"/test/file_{i}.mp4")
            time.sleep(0.01)
        
        return True
        
    except Exception as e:
        print(f"Producer process error: {e}")
        return False


def consumer_process(control_name: str, expected_updates: int):
    """Consumer process for multiprocess testing."""
    try:
        control = ControlState(control_name)
        if not control.connect():
            return False
        
        updates_seen = 0
        start_time = time.time()
        last_file = ""
        
        while updates_seen < expected_updates and (time.time() - start_time) < 5.0:
            status = control.get_status()
            if status and status.current_file != last_file:
                last_file = status.current_file
                updates_seen += 1
            time.sleep(0.005)
        
        return updates_seen >= expected_updates
        
    except Exception as e:
        print(f"Consumer process error: {e}")
        return False


@unittest.skipUnless(SHARED_MEMORY_AVAILABLE, "Shared memory not available (requires Python 3.8+)")
class TestMultiprocessControlState(unittest.TestCase):
    """Test multiprocess coordination through control state."""
    
    def test_multiprocess_coordination(self):
        """Test basic multiprocess coordination."""
        control_name = f"test_mp_control_{os.getpid()}_{int(time.time() * 1000000)}"
        
        # Initialize in main process
        control_state = ControlState(control_name)
        self.assertTrue(control_state.initialize())
        
        try:
            num_updates = 5
            
            # Start consumer process
            consumer = mp.Process(
                target=consumer_process,
                args=(control_name, num_updates)
            )
            consumer.start()
            
            # Start producer process
            producer = mp.Process(
                target=producer_process,
                args=(control_name, num_updates)
            )
            producer.start()
            
            # Wait for completion
            producer.join(timeout=10)
            consumer.join(timeout=10)
            
            # Verify both processes completed successfully
            self.assertEqual(producer.exitcode, 0, "Producer should complete successfully")
            self.assertEqual(consumer.exitcode, 0, "Consumer should complete successfully")
            
        finally:
            control_state.cleanup()
    
    def test_event_synchronization(self):
        """Test event-based synchronization between processes."""
        control_name = f"test_event_sync_{os.getpid()}_{int(time.time() * 1000000)}"
        
        control_state = ControlState(control_name)
        self.assertTrue(control_state.initialize())
        
        try:
            # Test shutdown event across processes
            def shutdown_waiter(control_name):
                control = ControlState(control_name)
                control.connect()
                return control.wait_for_shutdown(timeout=2.0)
            
            def shutdown_signaler(control_name):
                time.sleep(0.5)  # Wait a bit then signal
                control = ControlState(control_name)
                control.connect()
                control.signal_shutdown()
                return True
            
            waiter = mp.Process(target=shutdown_waiter, args=(control_name,))
            signaler = mp.Process(target=shutdown_signaler, args=(control_name,))
            
            waiter.start()
            signaler.start()
            
            waiter.join(timeout=5)
            signaler.join(timeout=5)
            
            # Both should complete successfully
            self.assertEqual(waiter.exitcode, 0, "Waiter should receive shutdown signal")
            self.assertEqual(signaler.exitcode, 0, "Signaler should complete successfully")
            
        finally:
            control_state.cleanup()


if __name__ == '__main__':
    # Configure logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    unittest.main(verbosity=2)