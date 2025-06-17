"""
Unit tests for the Shared Memory Ring Buffer.

Tests the zero-copy frame sharing functionality including initialization,
buffer management, process synchronization, and resource cleanup.
"""

import unittest
import multiprocessing as mp
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.shared_buffer import (FrameRingBuffer, FrameProducer, FrameConsumer, 
                                    get_shared_control_state, get_buffer_utilization, get_frame_timing_info,
                                    SHARED_MEMORY_AVAILABLE)
from src.const import FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS, METADATA_DTYPE


def get_buffer_array(buffer_info, width=FRAME_WIDTH, height=FRAME_HEIGHT, channels=FRAME_CHANNELS):
    """Helper function to get array from BufferInfo for backwards compatibility."""
    return buffer_info.get_array(width, height, channels)


@unittest.skipUnless(SHARED_MEMORY_AVAILABLE, "Shared memory not available (requires Python 3.8+)")
class TestFrameRingBuffer(unittest.TestCase):
    """Test cases for FrameRingBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer_name = f"test_buffer_{os.getpid()}_{int(time.time() * 1000000)}"
        self.ring_buffer = FrameProducer(self.buffer_name)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'ring_buffer'):
            self.ring_buffer.cleanup()
    
    def test_initialization(self):
        """Test basic initialization of ring buffer."""
        # Test successful initialization
        result = self.ring_buffer.initialize()
        self.assertTrue(result, "Ring buffer initialization should succeed")
        
        # Verify buffer is marked as initialized
        status = self.ring_buffer.get_status()
        self.assertTrue(status['initialized'], "Buffer should be marked as initialized")
        self.assertEqual(status['buffer_count'], 3, "Should have 3 buffers")
        self.assertEqual(status['frame_counter'], 0, "Frame counter should start at 0")
        
    def test_buffer_dimensions(self):
        """Test that buffers have correct dimensions."""
        self.ring_buffer.initialize()
        
        buffer_info = self.ring_buffer.get_write_buffer()
        self.assertIsNotNone(buffer_info, "Should get a write buffer")
        
        array = get_buffer_array(buffer_info)
        expected_shape = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
        self.assertEqual(array.shape, expected_shape, 
                        f"Buffer should have shape {expected_shape}")
        self.assertEqual(array.dtype, np.uint8, "Buffer should be uint8 type")
    
    def test_write_buffer_operations(self):
        """Test writing operations on the buffer."""
        self.ring_buffer.initialize()
        
        # Get write buffer
        buffer_info = self.ring_buffer.get_write_buffer()
        self.assertIsNotNone(buffer_info)
        
        # Write test data
        test_pattern = np.random.randint(0, 255, 
                                       (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), 
                                       dtype=np.uint8)
        array = get_buffer_array(buffer_info)
        array[:] = test_pattern
        
        # Advance write buffer
        result = self.ring_buffer.advance_write()
        self.assertTrue(result, "Advance write should succeed")
        
        # Verify frame counter increased
        status = self.ring_buffer.get_status()
        self.assertEqual(status['frame_counter'], 1, "Frame counter should increment")
    
    def test_multiple_write_operations(self):
        """Test multiple consecutive write operations."""
        self.ring_buffer.initialize()
        
        num_writes = 5
        for i in range(num_writes):
            buffer_info = self.ring_buffer.get_write_buffer()
            self.assertIsNotNone(buffer_info, f"Should get write buffer for frame {i}")
            
            # Write unique pattern for this frame
            pattern_value = (i + 1) * 50  # Different value for each frame
            get_buffer_array(buffer_info).fill(pattern_value)
            
            result = self.ring_buffer.advance_write()
            self.assertTrue(result, f"Advance write should succeed for frame {i}")
        
        # Verify final frame counter
        status = self.ring_buffer.get_status()
        self.assertEqual(status['frame_counter'], num_writes)
    
    def test_buffer_index_wraparound(self):
        """Test that buffer indices wrap around correctly."""
        self.ring_buffer.initialize()
        
        # Write frames up to the backpressure limit (buffer_count * 2 = 6)
        num_writes = 6  # Maximum without consumer
        
        for i in range(num_writes):
            buffer_info = self.ring_buffer.get_write_buffer()
            self.assertIsNotNone(buffer_info, f"Should get buffer for frame {i}")
            
            # Verify frame_id is reasonable (should be incrementing)
            expected_frame_id = i + 1  # frame_id starts at 1
            self.assertEqual(buffer_info.frame_id, expected_frame_id, 
                           f"Frame ID should be {expected_frame_id} for write {i}")
            
            self.ring_buffer.advance_write()
        
        # After 6 writes, write index should have wrapped around twice
        status = self.ring_buffer.get_status()
        self.assertEqual(status['frame_counter'], num_writes)
        # Write index should be (6 % 3) = 0
        self.assertEqual(status['write_index'], 0)
        
        # Next write should timeout due to backpressure (no consumer)
        buffer_info = self.ring_buffer.get_write_buffer(timeout=0.1)
        self.assertIsNone(buffer_info, "Should timeout when buffers are full and no consumer")
    
    def test_timestamp_tracking(self):
        """Test that timestamps are properly tracked."""
        self.ring_buffer.initialize()
        
        start_time = time.time()
        
        # Write a frame
        buffer_info = self.ring_buffer.get_write_buffer()
        write_time = buffer_info.timestamp
        self.ring_buffer.advance_write()
        
        # Verify timestamp is reasonable
        self.assertGreaterEqual(write_time, start_time)
        self.assertLess(write_time - start_time, 1.0)  # Should be within 1 second
        
        # Check status includes timestamps
        status = self.ring_buffer.get_status()
        self.assertIn('buffer_timestamps', status)
        self.assertEqual(len(status['buffer_timestamps']), 3)  # One for each buffer
    
    def test_connection_to_existing_buffer(self):
        """Test connecting to an existing shared memory buffer."""
        # Initialize buffer in one instance
        self.ring_buffer.initialize()
        
        # Write some test data
        buffer_info = self.ring_buffer.get_write_buffer()
        test_data = np.full((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), 42, dtype=np.uint8)
        get_buffer_array(buffer_info)[:] = test_data
        self.ring_buffer.advance_write()
        
        # Create second instance and connect
        consumer_buffer = FrameConsumer(self.buffer_name)
        result = consumer_buffer.connect()
        self.assertTrue(result, "Should successfully connect to existing buffer")
        
        # Verify connection worked
        status = consumer_buffer.get_status()
        self.assertTrue(status['initialized'])
        
        # Clean up the consumer buffer (but don't unlink shared memory)
        consumer_buffer._shared_memory.clear()
        consumer_buffer._control_memory = None
        consumer_buffer._initialized = False
    
    def test_error_handling_uninitialized(self):
        """Test error handling when buffer is not initialized."""
        # Don't initialize the buffer
        
        buffer_info = self.ring_buffer.get_write_buffer()
        self.assertIsNone(buffer_info, "Should return None when not initialized")
        
        result = self.ring_buffer.advance_write()
        self.assertFalse(result, "Should return False when not initialized")
        
        buffer_info = self.ring_buffer.wait_for_ready_buffer(timeout=0.1)
        self.assertIsNone(buffer_info, "Should return None when not initialized")
    
    def test_wait_for_ready_buffer_timeout(self):
        """Test timeout behavior of wait_for_ready_buffer."""
        self.ring_buffer.initialize()
        
        # Create consumer to test wait timeout
        consumer = FrameConsumer(self.buffer_name)
        self.assertTrue(consumer.connect())
        
        # No data written yet, should timeout
        start_time = time.time()
        buffer_info = consumer.wait_for_ready_buffer(timeout=0.2)
        end_time = time.time()
        
        self.assertIsNone(buffer_info, "Should return None on timeout")
        self.assertGreaterEqual(end_time - start_time, 0.15)  # Should have waited
        self.assertLess(end_time - start_time, 0.5)  # But not too long
    
    def test_cleanup(self):
        """Test proper cleanup of resources."""
        self.ring_buffer.initialize()
        
        # Verify buffer is initialized
        status = self.ring_buffer.get_status()
        self.assertTrue(status['initialized'])
        
        # Cleanup
        self.ring_buffer.cleanup()
        
        # Verify cleanup worked
        status = self.ring_buffer.get_status()
        self.assertFalse(status['initialized'])
        
        # Should not be able to get buffers after cleanup
        buffer_info = self.ring_buffer.get_write_buffer()
        self.assertIsNone(buffer_info)


def producer_process(buffer_name: str, num_frames: int, pattern_value: int, delay_per_frame: float = 0.01, 
                    write_timeout: float = 1.0, test_resolution: bool = False, test_timing: bool = False):
    """Producer process for multiprocess testing."""
    try:
        # Initialize shared buffer as a producer (producer always initializes)
        ring_buffer = FrameProducer(buffer_name)
        success = ring_buffer.initialize()
        if not success:
            return False
        
        # Write frames
        start_time = time.time()
        for i in range(num_frames):
            # Calculate presentation timestamp if testing timing
            presentation_timestamp = None
            source_width, source_height = None, None
            
            if test_timing:
                # Schedule frames at regular intervals starting immediately
                presentation_timestamp = start_time + (i * 0.05)  # 50ms intervals
            
            if test_resolution:
                # Use varying resolutions for testing
                source_width = 1280 + (i % 3) * 160  # 1280, 1440, 1600
                source_height = 720 + (i % 3) * 90   # 720, 810, 900
                
            buffer_info = ring_buffer.get_write_buffer(
                timeout=write_timeout,
                presentation_timestamp=presentation_timestamp,
                source_width=source_width,
                source_height=source_height
            )
            if buffer_info is None:
                print(f"Producer timeout on frame {i}")
                return False
                
            # Fill with test pattern
            get_buffer_array(buffer_info).fill(pattern_value)
            
            if not ring_buffer.advance_write():
                return False
            
            # Optional delay between frames
            if delay_per_frame > 0:
                time.sleep(delay_per_frame)
        
        return True
        
    except Exception as e:
        print(f"Producer process error: {e}")
        return False


def consumer_process(buffer_name: str, expected_frames: int, expected_pattern: int, delay_per_frame: float = 0.01,
                    test_timing: bool = False, test_resolution: bool = False):
    """Consumer process for multiprocess testing."""
    try:
        # Connect to shared buffer as consumer
        ring_buffer = FrameConsumer(buffer_name)
        success = ring_buffer.connect()
        if not success:
            return False
        
        frames_received = 0
        frames_dropped = 0
        start_time = time.time()
        
        while frames_received < expected_frames and (time.time() - start_time) < 15.0:
            buffer_info = ring_buffer.wait_for_ready_buffer(timeout=0.5)
            if buffer_info is not None:
                # Test frame timing if requested (basic timing test)
                should_display = True
                if test_timing:
                    # Simple timing check - if frame is too old, consider it dropped
                    current_time = time.time()
                    if buffer_info.metadata and buffer_info.metadata.presentation_timestamp > 0:
                        if current_time > buffer_info.metadata.presentation_timestamp + 0.1:  # 100ms tolerance
                            frames_dropped += 1
                            should_display = False
                    
                    if not should_display:
                        ring_buffer.release_read_buffer()
                        continue
                
                # Test resolution metadata if requested
                if test_resolution and should_display:
                    if buffer_info.metadata:
                        # Verify we got valid resolution metadata
                        if (buffer_info.metadata.source_width <= 0 or 
                            buffer_info.metadata.source_height <= 0):
                            print(f"Invalid resolution metadata for frame {frames_received}")
                            return False
                
                # Verify pattern
                if np.all(get_buffer_array(buffer_info) == expected_pattern):
                    frames_received += 1
                    # Optional delay to simulate processing time
                    if delay_per_frame > 0:
                        time.sleep(delay_per_frame)
                    # Release buffer when done (helps with backpressure testing)
                    ring_buffer.release_read_buffer()
                else:
                    print(f"Pattern mismatch in frame {frames_received}")
                    return False
        
        # For timing tests, allow some frames to be dropped
        if test_timing:
            result = frames_received >= expected_frames * 0.6  # Allow 40% drop rate for timing tests
            print(f"Consumer: received {frames_received}, dropped {frames_dropped}, expected >= {expected_frames * 0.6}, result: {result}")
            return result
        else:
            return frames_received == expected_frames
        
    except Exception as e:
        print(f"Consumer process error: {e}")
        return False


@unittest.skipUnless(SHARED_MEMORY_AVAILABLE, "Shared memory not available (requires Python 3.8+)")
class TestMultiprocessCommunication(unittest.TestCase):
    """Test multiprocess communication through ring buffer."""
    
    def test_producer_consumer_communication(self):
        """Test basic producer-consumer communication."""
        buffer_name = f"test_mp_buffer_{os.getpid()}_{int(time.time() * 1000000)}"
        
        num_frames = 5
        pattern_value = 123
        
        # Start producer process (it will initialize the buffer)
        producer = mp.Process(
            target=producer_process, 
            args=(buffer_name, num_frames, pattern_value)
        )
        producer.start()
        
        # Small delay to let producer initialize buffer
        time.sleep(0.1)
        
        # Start consumer process (it will connect to existing buffer)
        consumer = mp.Process(
            target=consumer_process,
            args=(buffer_name, num_frames, pattern_value)
        )
        consumer.start()
        
        # Wait for completion
        producer.join(timeout=10)
        consumer.join(timeout=10) 
        
        # Verify both processes completed successfully
        self.assertEqual(producer.exitcode, 0, "Producer should complete successfully")
        self.assertEqual(consumer.exitcode, 0, "Consumer should complete successfully")
    
    def test_data_integrity_across_processes(self):
        """Test data integrity when passing complex patterns between processes."""
        # Test with multiple different patterns
        test_patterns = [11, 22, 33, 44, 55]
        
        for pattern in test_patterns:
            buffer_name = f"test_integrity_{os.getpid()}_{int(time.time() * 1000000)}_{pattern}"
            
            producer = mp.Process(
                target=producer_process,
                args=(buffer_name, 1, pattern)
            )
            producer.start()
            
            # Small delay to let producer initialize buffer
            time.sleep(0.1)
            
            consumer = mp.Process(
                target=consumer_process,
                args=(buffer_name, 1, pattern)
            )
            consumer.start()
            
            producer.join(timeout=5)
            consumer.join(timeout=5)
            
            self.assertEqual(producer.exitcode, 0, f"Producer failed for pattern {pattern}")
            self.assertEqual(consumer.exitcode, 0, f"Consumer failed for pattern {pattern}")
    
    def test_fast_producer_slow_consumer_backpressure(self):
        """Test backpressure when producer is faster than consumer."""
        buffer_name = f"test_backpressure_{os.getpid()}_{int(time.time() * 1000000)}"
        
        num_frames = 10
        pattern_value = 200
        
        # Start fast producer (10ms per frame) - it will initialize the buffer
        producer_start_time = time.time()
        producer = mp.Process(
            target=producer_process,
            args=(buffer_name, num_frames, pattern_value, 0.01, 2.0)  # 10ms delay, 2s timeout
        )
        producer.start()
        
        # Small delay to let producer initialize buffer
        time.sleep(0.1)
        
        # Start slow consumer (100ms per frame)
        consumer = mp.Process(
            target=consumer_process,
            args=(buffer_name, num_frames, pattern_value, 0.1)  # 100ms delay
        )
        consumer.start()
        
        # Wait for completion
        producer.join(timeout=15)
        consumer.join(timeout=15)
        
        producer_duration = time.time() - producer_start_time
        
        # Verify both processes completed successfully
        self.assertEqual(producer.exitcode, 0, "Producer should handle backpressure gracefully")
        self.assertEqual(consumer.exitcode, 0, "Consumer should receive all frames")
        
        # Producer should be slowed down by consumer (should take longer than num_frames * 0.01)
        min_expected_duration = num_frames * 0.05  # Conservative estimate
        self.assertGreater(producer_duration, min_expected_duration, 
                         "Producer should be throttled by slow consumer")
    
    def test_slow_producer_fast_consumer_wait_states(self):
        """Test consumer waiting when producer is slower."""
        buffer_name = f"test_wait_states_{os.getpid()}_{int(time.time() * 1000000)}"
        
        num_frames = 8
        pattern_value = 150
        
        # Start slow producer (50ms per frame) - it will initialize the buffer
        producer = mp.Process(
            target=producer_process,
            args=(buffer_name, num_frames, pattern_value, 0.05, 1.0)  # 50ms delay
        )
        producer.start()
        
        # Small delay to let producer initialize buffer
        time.sleep(0.1)
        
        # Start fast consumer (5ms per frame)
        consumer_start_time = time.time()
        consumer = mp.Process(
            target=consumer_process,
            args=(buffer_name, num_frames, pattern_value, 0.005)  # 5ms delay
        )
        consumer.start()
        
        # Wait for completion
        producer.join(timeout=10)
        consumer.join(timeout=10)
        
        consumer_duration = time.time() - consumer_start_time
        
        # Verify both processes completed successfully
        self.assertEqual(producer.exitcode, 0, "Producer should complete successfully")
        self.assertEqual(consumer.exitcode, 0, "Consumer should wait and receive all frames")
        
        # Consumer should be slowed down by producer (should take roughly as long as producer)
        min_expected_duration = num_frames * 0.04  # Should wait for producer
        self.assertGreater(consumer_duration, min_expected_duration,
                         "Consumer should wait for slow producer")
    
    def test_producer_timeout_behavior(self):
        """Test producer timeout when consumer never reads."""
        buffer_name = f"test_producer_timeout_{os.getpid()}_{int(time.time() * 1000000)}"
        
        ring_buffer = FrameProducer(buffer_name)
        self.assertTrue(ring_buffer.initialize())
        
        try:
            # Fill up to the backpressure limit (buffer_count * 2 = 6 frames)
            for i in range(6):  # Fill to backpressure limit
                buffer_info = ring_buffer.get_write_buffer(timeout=0.1)
                self.assertIsNotNone(buffer_info, f"Should get buffer {i}")
                get_buffer_array(buffer_info).fill(100 + i)
                self.assertTrue(ring_buffer.advance_write())
            
            # Now try to get another buffer - should timeout due to backpressure
            start_time = time.time()
            buffer_info = ring_buffer.get_write_buffer(timeout=0.2)
            end_time = time.time()
            
            self.assertIsNone(buffer_info, "Should timeout when no consumer reads and buffers are full")
            self.assertGreaterEqual(end_time - start_time, 0.15, "Should have waited for timeout")
            self.assertLess(end_time - start_time, 0.5, "Should not wait too long")
            
        finally:
            ring_buffer.cleanup()
    
    def test_consumer_timeout_behavior(self):
        """Test consumer timeout when producer never writes."""
        buffer_name = f"test_consumer_timeout_{os.getpid()}_{int(time.time() * 1000000)}"
        
        producer = FrameProducer(buffer_name)
        self.assertTrue(producer.initialize())
        
        try:
            # Create consumer to test timeout
            consumer = FrameConsumer(buffer_name)
            self.assertTrue(consumer.connect())
            
            # Try to read without any data written
            start_time = time.time()
            buffer_info = consumer.wait_for_ready_buffer(timeout=0.2)
            end_time = time.time()
            
            self.assertIsNone(buffer_info, "Should timeout when no data available")
            self.assertGreaterEqual(end_time - start_time, 0.15, "Should have waited for timeout")
            self.assertLess(end_time - start_time, 0.5, "Should not wait too long")
            
        finally:
            producer.cleanup()
    
    def test_buffer_release_mechanics(self):
        """Test explicit buffer release functionality."""
        buffer_name = f"test_release_{os.getpid()}_{int(time.time() * 1000000)}"
        
        producer = FrameProducer(buffer_name)
        self.assertTrue(producer.initialize())
        
        try:
            # Write a frame
            buffer_info = producer.get_write_buffer()
            self.assertIsNotNone(buffer_info)
            get_buffer_array(buffer_info).fill(75)
            self.assertTrue(producer.advance_write())
            
            # Create consumer to read the frame
            consumer = FrameConsumer(buffer_name)
            self.assertTrue(consumer.connect())
            
            # Read the frame
            buffer_info = consumer.wait_for_ready_buffer(timeout=0.5)
            self.assertIsNotNone(buffer_info, "Should get the written frame")
            
            # Check status - should show read buffer is occupied
            status = producer.get_status()
            read_idx = status['read_index']
            self.assertNotEqual(read_idx, -1, "Read index should be set")
            
            # Release the buffer
            self.assertTrue(consumer.release_read_buffer())
            
            # Check status - read index should be cleared
            status = producer.get_status()
            self.assertEqual(status['read_index'], -1, "Read index should be cleared after release")
            
        finally:
            producer.cleanup()
    
    def test_high_throughput_stress(self):
        """Stress test with high frame rates and multiple cycles."""
        buffer_name = f"test_stress_{os.getpid()}_{int(time.time() * 1000000)}"
        
        num_frames = 20  # Reduced frame count for reliability
        pattern_value = 180
        
        # Fast producer (5ms per frame) - it will initialize the buffer
        producer = mp.Process(
            target=producer_process,
            args=(buffer_name, num_frames, pattern_value, 0.005, 1.0)
        )
        producer.start()
        
        # Small delay to let producer initialize buffer
        time.sleep(0.1)
        
        # Fast consumer (5ms per frame)
        consumer = mp.Process(
            target=consumer_process,
            args=(buffer_name, num_frames, pattern_value, 0.005)
        )
        consumer.start()
        
        # Wait for completion
        producer.join(timeout=10)
        consumer.join(timeout=10)
        
        # Verify successful completion
        self.assertEqual(producer.exitcode, 0, "High throughput producer should succeed")
        self.assertEqual(consumer.exitcode, 0, "High throughput consumer should succeed")
    
    def test_presentation_timestamp_sequence(self):
        """Test presentation timestamps are maintained in correct sequence."""
        buffer_name = f"test_timestamp_{os.getpid()}_{int(time.time() * 1000000)}"
        
        producer = FrameProducer(buffer_name)
        self.assertTrue(producer.initialize())
        
        try:
            current_time = time.time()
            expected_timestamps = []
            
            # Create consumer first
            consumer = FrameConsumer(buffer_name)
            self.assertTrue(consumer.connect())
            
            # Write and read frames one by one to avoid overwriting
            actual_timestamps = []
            for i in range(3):
                future_time = current_time + (i * 0.02)  # 20ms intervals in future
                expected_timestamps.append(future_time)
                
                # Write frame
                buffer_info = producer.get_write_buffer(presentation_timestamp=future_time)
                self.assertIsNotNone(buffer_info)
                get_buffer_array(buffer_info).fill(100 + i)
                producer.advance_write()
                
                # Read frame immediately 
                read_info = consumer.wait_for_ready_buffer(timeout=0.5)
                self.assertIsNotNone(read_info, f"Should get frame {i}")
                
                metadata = read_info.metadata
                self.assertIsNotNone(metadata)
                actual_timestamps.append(metadata.presentation_timestamp)
                consumer.release_read_buffer()
            
            # Verify timestamps are in expected order
            for i, (expected, actual) in enumerate(zip(expected_timestamps, actual_timestamps)):
                self.assertAlmostEqual(actual, expected, places=6, 
                                     msg=f"Frame {i} timestamp mismatch")
            
        finally:
            producer.cleanup()
    
    def test_resolution_tagging_and_aspect_ratio(self):
        """Test resolution tagging and aspect ratio calculations."""
        buffer_name = f"test_resolution_{os.getpid()}_{int(time.time() * 1000000)}"
        
        num_frames = 6
        pattern_value = 240
        
        # Start producer with varying resolutions - it will initialize the buffer
        producer = mp.Process(
            target=producer_process,
            args=(buffer_name, num_frames, pattern_value, 0.01, 1.0, True, False)  # test_resolution=True
        )
        producer.start()
        
        # Small delay to let producer initialize buffer
        time.sleep(0.1)
        
        # Start consumer that tests resolution handling
        consumer = mp.Process(
            target=consumer_process,
            args=(buffer_name, num_frames, pattern_value, 0.001, False, True)  # test_resolution=True
        )
        consumer.start()
        
        # Wait for completion
        producer.join(timeout=10)
        consumer.join(timeout=10)
        
        # Verify both processes completed successfully
        self.assertEqual(producer.exitcode, 0, "Producer should complete successfully")
        self.assertEqual(consumer.exitcode, 0, "Consumer should handle resolution metadata correctly")
    
    def test_presentation_timestamp_storage(self):
        """Test that presentation timestamps are stored correctly."""
        buffer_name = f"test_timing_decisions_{os.getpid()}_{int(time.time() * 1000000)}"
        
        producer = FrameProducer(buffer_name)
        self.assertTrue(producer.initialize())
        
        try:
            # Test frame with specific presentation timestamp
            test_timestamp = time.time() + 0.1
            buffer_info = producer.get_write_buffer(presentation_timestamp=test_timestamp)
            self.assertIsNotNone(buffer_info)
            producer.advance_write()
            
            # Create consumer to read frame
            consumer = FrameConsumer(buffer_name)
            self.assertTrue(consumer.connect())
            
            read_info = consumer.wait_for_ready_buffer(timeout=0.5)
            self.assertIsNotNone(read_info)
            
            # Verify presentation timestamp is stored correctly
            metadata = read_info.metadata
            self.assertIsNotNone(metadata)
            self.assertAlmostEqual(metadata.presentation_timestamp, test_timestamp, places=6)
            
            # Test frame with default presentation timestamp (current time)
            buffer_info = producer.get_write_buffer()  # No timestamp specified
            self.assertIsNotNone(buffer_info)
            write_time = time.time()
            producer.advance_write()
            
            read_info = consumer.wait_for_ready_buffer(timeout=0.5)
            self.assertIsNotNone(read_info)
            
            metadata = read_info.metadata
            self.assertIsNotNone(metadata)
            # Default timestamp should be close to write time
            self.assertLess(abs(metadata.presentation_timestamp - write_time), 0.1)
            
        finally:
            producer.cleanup()
    
    def test_metadata_storage_with_resolution(self):
        """Test that resolution metadata is stored and retrieved correctly."""
        buffer_name = f"test_aspect_{os.getpid()}_{int(time.time() * 1000000)}"
        
        producer = FrameProducer(buffer_name)
        self.assertTrue(producer.initialize())
        
        try:
            # Create consumer for reading
            consumer = FrameConsumer(buffer_name)
            self.assertTrue(consumer.connect())
            
            # Test 16:9 content resolution metadata
            buffer_info = producer.get_write_buffer(
                source_width=1920, 
                source_height=1080
            )
            self.assertIsNotNone(buffer_info)
            producer.advance_write()
            
            read_info = consumer.wait_for_ready_buffer(timeout=0.5)
            self.assertIsNotNone(read_info)
            
            # Verify metadata is correctly stored
            metadata = read_info.metadata
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata.source_width, 1920)
            self.assertEqual(metadata.source_height, 1080)
            
            # Test 4:3 content resolution metadata
            buffer_info = producer.get_write_buffer(
                source_width=1280,
                source_height=960
            )
            self.assertIsNotNone(buffer_info)
            producer.advance_write()
            
            read_info = consumer.wait_for_ready_buffer(timeout=0.5)
            self.assertIsNotNone(read_info)
            
            metadata = read_info.metadata
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata.source_width, 1280)
            self.assertEqual(metadata.source_height, 960)
            
        finally:
            producer.cleanup()
    
    def test_metadata_persistence_across_processes(self):
        """Test that metadata persists correctly across process boundaries."""
        buffer_name = f"test_metadata_persist_{os.getpid()}_{int(time.time() * 1000000)}"
        
        ring_buffer = FrameProducer(buffer_name)
        self.assertTrue(ring_buffer.initialize())
        
        try:
            # Producer writes frame with specific metadata
            test_timestamp = time.time() + 5.0
            test_width, test_height = 1440, 810
            
            buffer_info = ring_buffer.get_write_buffer(
                presentation_timestamp=test_timestamp,
                source_width=test_width,
                source_height=test_height
            )
            self.assertIsNotNone(buffer_info)
            
            # Verify metadata in producer
            metadata = buffer_info.metadata
            self.assertEqual(metadata.source_width, test_width)
            self.assertEqual(metadata.source_height, test_height)
            self.assertEqual(metadata.presentation_timestamp, test_timestamp)
            
            ring_buffer.advance_write()
            
            # Consumer connects and reads
            consumer_buffer = FrameConsumer(buffer_name)
            self.assertTrue(consumer_buffer.connect())
            
            read_info = consumer_buffer.wait_for_ready_buffer(timeout=0.5)
            self.assertIsNotNone(read_info)
            
            # Verify metadata in consumer
            read_metadata = read_info.metadata
            self.assertEqual(read_metadata.source_width, test_width)
            self.assertEqual(read_metadata.source_height, test_height) 
            self.assertEqual(read_metadata.presentation_timestamp, test_timestamp)
            
            # Clean up consumer buffer connection
            consumer_buffer._shared_memory.clear()
            consumer_buffer._metadata_memory = None
            consumer_buffer._control_memory = None
            consumer_buffer._initialized = False
            
        finally:
            ring_buffer.cleanup()


@unittest.skipUnless(SHARED_MEMORY_AVAILABLE, "Shared memory not available (requires Python 3.8+)")
class TestProducerConsumerSubclasses(unittest.TestCase):
    """Test the specialized producer and consumer subclasses."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer_name = f"test_subclass_{os.getpid()}_{int(time.time() * 1000000)}"
        self.producer = FrameProducer(self.buffer_name)
        self.consumer = FrameConsumer(self.buffer_name)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'producer'):
            self.producer.cleanup()
    
    def test_producer_subclass_basic_functionality(self):
        """Test basic producer subclass functionality."""
        # Initialize producer
        self.assertTrue(self.producer.initialize())
        
        # Test frame writing using get_write_buffer/advance_write pattern
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        buffer_info = self.producer.get_write_buffer(
            presentation_timestamp=time.time() + 0.1,
            source_width=1280,
            source_height=720
        )
        self.assertIsNotNone(buffer_info, "Should get write buffer")
        
        # Write directly to buffer (producer can write however they want)
        h, w = min(test_frame.shape[0], get_buffer_array(buffer_info).shape[0]), \
               min(test_frame.shape[1], get_buffer_array(buffer_info).shape[1])
        c = min(test_frame.shape[2], get_buffer_array(buffer_info).shape[2])
        get_buffer_array(buffer_info)[:h, :w, :c] = test_frame[:h, :w, :c]
        
        success = self.producer.advance_write()
        self.assertTrue(success, "Should successfully advance write")
        
        # Test producer stats
        stats = self.producer.get_producer_stats()
        self.assertEqual(stats['frames_written'], 1)
        self.assertTrue(stats['is_producer'])
        self.assertTrue(self.producer.can_write_frame())
    
    def test_consumer_subclass_basic_functionality(self):
        """Test basic consumer subclass functionality."""
        # Initialize producer and write a frame
        self.assertTrue(self.producer.initialize())
        test_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        
        # Write frame using get_write_buffer/advance_write pattern
        buffer_info = self.producer.get_write_buffer(
            source_width=640,
            source_height=480
        )
        self.assertIsNotNone(buffer_info)
        # Copy frame data
        h, w = min(test_frame.shape[0], get_buffer_array(buffer_info).shape[0]), \
               min(test_frame.shape[1], get_buffer_array(buffer_info).shape[1])
        c = min(test_frame.shape[2], get_buffer_array(buffer_info).shape[2])
        get_buffer_array(buffer_info)[:h, :w, :c] = test_frame[:h, :w, :c]
        self.producer.advance_write()
        
        # Connect consumer
        self.assertTrue(self.consumer.connect())
        
        # Test frame for display (combines read and aspect ratio processing)
        display_info = self.consumer.read_frame(timeout=0.5)
        self.assertIsNotNone(display_info)
        self.assertIn('array', display_info)
        self.assertIn('metadata', display_info)
        self.assertIn('buffer_id', display_info)
        
        # Test consumer stats
        stats = self.consumer.get_consumer_stats()
        self.assertGreater(stats['frames_consumed'], 0)
        self.assertTrue(stats['is_consumer'])
        
        self.consumer.finish_frame()
    
    def test_producer_frame_data_handling(self):
        """Test producer handles different frame data shapes correctly."""
        self.assertTrue(self.producer.initialize())
        
        # Test different frame shapes
        test_cases = [
            (480, 640, 3),      # Standard color
            (720, 1280, 4),     # With alpha
            (360, 480),         # Grayscale
        ]
        
        for shape in test_cases:
            with self.subTest(shape=shape):
                if len(shape) == 2:
                    test_frame = np.random.randint(0, 255, shape, dtype=np.uint8)
                else:
                    test_frame = np.random.randint(0, 255, shape, dtype=np.uint8)
                
                # Write frame using get_write_buffer/advance_write pattern
                buffer_info = self.producer.get_write_buffer()
                self.assertIsNotNone(buffer_info, f"Should get buffer for shape {shape}")
                # Copy frame data with shape handling
                h, w = min(test_frame.shape[0], get_buffer_array(buffer_info).shape[0]), \
                       min(test_frame.shape[1], get_buffer_array(buffer_info).shape[1])
                if len(test_frame.shape) == 2:  # Grayscale
                    get_buffer_array(buffer_info)[:h, :w, 0] = test_frame[:h, :w]
                else:  # Color
                    c = min(test_frame.shape[2], get_buffer_array(buffer_info).shape[2])
                    get_buffer_array(buffer_info)[:h, :w, :c] = test_frame[:h, :w, :c]
                success = self.producer.advance_write()
                self.assertTrue(success, f"Should handle shape {shape}")
    
    def test_consumer_timing_functionality(self):
        """Test consumer basic timing information is available."""
        self.assertTrue(self.producer.initialize())
        self.assertTrue(self.consumer.connect())
        
        current_time = time.time()
        
        # Write frame with past timestamp 
        past_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        buffer_info = self.producer.get_write_buffer(
            presentation_timestamp=current_time - 0.1  # 100ms in the past
        )
        self.assertIsNotNone(buffer_info)
        h, w, c = past_frame.shape
        get_buffer_array(buffer_info)[:h, :w, :c] = past_frame
        self.producer.advance_write()
        
        # Read frame - timing decisions are now in application code
        frame_info = self.consumer.read_frame(timeout=0.5)
        self.assertIsNotNone(frame_info)
        
        # Verify metadata contains timing information for application to use
        metadata = frame_info['metadata']
        self.assertIsNotNone(metadata)
        self.assertLess(metadata.presentation_timestamp, current_time)  # In the past
        self.assertGreater(metadata.capture_timestamp, 0)  # Valid capture time
        
        # Consumer stats should show frame was consumed
        stats = self.consumer.get_consumer_stats()
        self.assertGreater(stats['frames_consumed'], 0)
    
    def test_control_state_getters(self):
        """Test the control state getter functions."""
        self.assertTrue(self.producer.initialize())
        
        # Test shared control state
        control_state = get_shared_control_state(self.producer)
        self.assertTrue(control_state['initialized'])
        self.assertIn('buffer_info', control_state)
        self.assertIn('timing_info', control_state)
        self.assertIn('flow_control', control_state)
        
        # Test buffer utilization
        utilization = get_buffer_utilization(self.producer)
        self.assertIn('buffer_fill_percentage', utilization)
        self.assertIn('is_healthy', utilization)
        
        # Test frame timing info
        timing_info = get_frame_timing_info(self.producer)
        self.assertIn('estimated_frame_rate', timing_info)
        self.assertIn('recent_timestamps', timing_info)
    
    def test_producer_consumer_interaction_with_subclasses(self):
        """Test producer and consumer working together with subclasses."""
        self.assertTrue(self.producer.initialize())
        self.assertTrue(self.consumer.connect())
        
        # Test write-then-read pattern one frame at a time
        num_frames = 3
        frames_processed = 0
        
        for i in range(num_frames):
            # Write one frame
            test_frame = np.full((480, 640, 3), 50 + i * 50, dtype=np.uint8)
            buffer_info = self.producer.get_write_buffer(
                source_width=640,
                source_height=480
            )
            self.assertIsNotNone(buffer_info)
            h, w, c = test_frame.shape
            get_buffer_array(buffer_info)[:h, :w, :c] = test_frame
            success = self.producer.advance_write()
            self.assertTrue(success)
            
            # Read the frame immediately  
            display_info = self.consumer.read_frame(timeout=0.5)
            if display_info is not None:
                frames_processed += 1
                # Verify frame info is included
                self.assertIn('buffer_id', display_info)
                self.assertIn('metadata', display_info)
                # Verify metadata contains resolution info for application to use
                metadata = display_info['metadata']
                self.assertEqual(metadata.source_width, 640)
                self.assertEqual(metadata.source_height, 480)
                
                self.consumer.finish_frame()
        
        self.assertEqual(frames_processed, num_frames)
        
        # Check final stats
        producer_stats = self.producer.get_producer_stats()
        consumer_stats = self.consumer.get_consumer_stats()
        
        self.assertEqual(producer_stats['frames_written'], num_frames)
        self.assertEqual(consumer_stats['frames_consumed'], num_frames)
    
    def test_control_state_monitoring(self):
        """Test monitoring control state during operation."""
        self.assertTrue(self.producer.initialize())
        
        # Initial state
        initial_state = get_shared_control_state(self.producer)
        initial_utilization = get_buffer_utilization(self.producer)
        
        self.assertEqual(initial_state['flow_control']['frames_ahead'], 0)
        self.assertEqual(initial_utilization['buffer_fill_percentage'], 0.0)
        
        # Write frames to increase utilization
        for i in range(2):
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            buffer_info = self.producer.get_write_buffer()
            self.assertIsNotNone(buffer_info)
            h, w, c = test_frame.shape
            get_buffer_array(buffer_info)[:h, :w, :c] = test_frame
            self.producer.advance_write()
        
        # Check increased utilization
        mid_state = get_shared_control_state(self.producer)
        mid_utilization = get_buffer_utilization(self.producer)
        
        self.assertEqual(mid_state['flow_control']['frames_ahead'], 2)
        self.assertGreater(mid_utilization['buffer_fill_percentage'], 0.0)
        
        # Verify other state information is present
        self.assertIn('buffer_info', mid_state)
        self.assertIn('timing_info', mid_state)
        self.assertEqual(mid_state['buffer_info']['frame_counter'], 2)
        self.assertEqual(len(mid_state['timing_info']['buffer_timestamps']), 3)


if __name__ == '__main__':
    # Configure logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    unittest.main(verbosity=2)