"""
Unit tests for PerformanceTiming class.

This module provides comprehensive testing for the PerformanceTiming framework,
including basic timing, nested sections, memory bandwidth calculations,
FLOPS analysis, error handling, and export functionality.
"""

import csv
import logging
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from src.utils.performance_timing import (
    PerformanceTiming,
    TimingSection,
    TimingSectionContext,
)


class TestPerformanceTiming(unittest.TestCase):
    """Test cases for PerformanceTiming class."""

    def setUp(self):
        """Set up test fixtures."""
        self.timing = PerformanceTiming("TestModule")
        self.logger = Mock(spec=logging.Logger)

    def tearDown(self):
        """Clean up after tests."""
        self.timing.reset()

    def test_initialization(self):
        """Test PerformanceTiming initialization."""
        timing = PerformanceTiming("TestModule", enable_gpu_timing=False)

        self.assertEqual(timing.module_name, "TestModule")
        self.assertFalse(timing.enable_gpu_timing)
        self.assertEqual(len(timing._sections), 0)
        self.assertEqual(len(timing._section_stack), 0)
        self.assertIsNone(timing._current_section)
        self.assertEqual(timing._error_count, 0)

    def test_basic_timing(self):
        """Test basic start/stop timing functionality."""
        # Start timing
        result = self.timing.start("test_section")
        self.assertTrue(result)

        # Simulate some work
        time.sleep(0.01)

        # Stop timing
        result = self.timing.stop("test_section")
        self.assertTrue(result)

        # Verify section was created and timed
        self.assertIn("test_section", self.timing._sections)
        section = self.timing._sections["test_section"]
        self.assertGreater(section.duration, 0.005)  # Should be at least 5ms
        self.assertEqual(section.occurrence_count, 1)

    def test_metrics_on_start(self):
        """Test providing metrics on start() method."""
        self.timing.start("test_section", read=1000, written=500, flops=2000)
        time.sleep(0.01)
        self.timing.stop("test_section")

        section = self.timing._sections["test_section"]
        self.assertEqual(section.read_bytes, 1000)
        self.assertEqual(section.written_bytes, 500)
        self.assertEqual(section.flops, 2000)

    def test_metrics_on_stop(self):
        """Test providing metrics on stop() method."""
        self.timing.start("test_section")
        time.sleep(0.01)
        self.timing.stop("test_section", read=1000, written=500, flops=2000)

        section = self.timing._sections["test_section"]
        self.assertEqual(section.read_bytes, 1000)
        self.assertEqual(section.written_bytes, 500)
        self.assertEqual(section.flops, 2000)

    def test_metrics_on_both_start_and_stop(self):
        """Test providing metrics on both start() and stop() methods."""
        self.timing.start("test_section", read=500, written=250, flops=1000)
        time.sleep(0.01)
        self.timing.stop("test_section", read=500, written=250, flops=1000)

        section = self.timing._sections["test_section"]
        self.assertEqual(section.read_bytes, 1000)  # Sum of both
        self.assertEqual(section.written_bytes, 500)  # Sum of both
        self.assertEqual(section.flops, 2000)  # Sum of both

    def test_nested_sections(self):
        """Test nested timing sections."""
        # Start outer section
        self.timing.start("outer_section")
        time.sleep(0.01)

        # Start inner section
        self.timing.start("inner_section")
        time.sleep(0.01)
        self.timing.stop("inner_section")

        # Stop outer section
        time.sleep(0.01)
        self.timing.stop("outer_section")

        # Verify nesting structure
        outer = self.timing._sections["outer_section"]
        inner = self.timing._sections["inner_section"]

        self.assertEqual(outer.depth, 0)
        self.assertEqual(inner.depth, 1)
        self.assertEqual(inner.parent, outer)
        self.assertIn(inner, outer.children)

        # Verify timing relationships
        self.assertGreater(outer.duration, inner.duration)

    def test_repeated_sections(self):
        """Test repeated execution of the same section."""
        durations = []

        for i in range(3):
            self.timing.start("repeated_section", flops=1000)
            time.sleep(0.01 + i * 0.005)  # Varying durations
            self.timing.stop("repeated_section")
            durations.append(self.timing._sections["repeated_section"].duration)

        section = self.timing._sections["repeated_section"]
        self.assertEqual(section.occurrence_count, 3)
        self.assertEqual(len(section.occurrence_durations), 3)
        self.assertEqual(len(section.occurrence_metrics), 3)

        # Verify first occurrence data is stored separately
        self.assertIsNotNone(section.first_occurrence_data)
        self.assertEqual(section.first_occurrence_data["flops"], 1000)

        # Verify all occurrences are tracked
        for i, duration in enumerate(section.occurrence_durations):
            self.assertAlmostEqual(duration, durations[i], places=3)

    def test_context_manager_basic(self):
        """Test context manager functionality."""
        with self.timing.section("context_section", read=1000, flops=2000):
            time.sleep(0.01)

        self.assertIn("context_section", self.timing._sections)
        section = self.timing._sections["context_section"]
        self.assertGreater(section.duration, 0.005)
        self.assertEqual(section.read_bytes, 1000)
        self.assertEqual(section.flops, 2000)

    def test_context_manager_with_additional_metrics(self):
        """Test context manager with additional metrics."""
        with self.timing.section("context_section", read=500) as ctx:
            time.sleep(0.01)
            ctx.add_memory(read=500, written=1000)

        section = self.timing._sections["context_section"]
        self.assertEqual(section.read_bytes, 1000)  # 500 + 500
        self.assertEqual(section.written_bytes, 1000)

    def test_context_manager_with_exception(self):
        """Test context manager handles exceptions properly."""
        try:
            with self.timing.section("exception_section"):
                time.sleep(0.01)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Section should still be properly closed
        self.assertIn("exception_section", self.timing._sections)
        section = self.timing._sections["exception_section"]
        self.assertGreater(section.duration, 0.005)

    def test_memory_transfer_tracking(self):
        """Test memory transfer tracking."""
        self.timing.start("transfer_section")
        self.timing.record_memory_transfer(100.0, "cpu_to_gpu")
        self.timing.record_memory_transfer(50.0, "gpu_to_cpu")
        self.timing.stop("transfer_section")

        section = self.timing._sections["transfer_section"]
        self.assertEqual(len(section.memory_transfers), 2)

        transfers = section.memory_transfers
        self.assertEqual(transfers[0]["size_mb"], 100.0)
        self.assertEqual(transfers[0]["direction"], "cpu_to_gpu")
        self.assertEqual(transfers[1]["size_mb"], 50.0)
        self.assertEqual(transfers[1]["direction"], "gpu_to_cpu")

        # Check global counters
        self.assertEqual(self.timing._total_gpu_memory_transfers, 2)
        self.assertEqual(self.timing._total_cpu_gpu_transfers, 2)

    def test_automatic_flops_calculation(self):
        """Test automatic FLOPS calculation for different operation types."""
        # Test einsum
        shape_info = {"A": (100, 100), "B": (100, 50)}
        self.timing.start("einsum_test", operation_type="einsum", shape_info=shape_info)
        self.timing.stop("einsum_test")

        section = self.timing._sections["einsum_test"]
        self.assertGreater(section.flops, 0)

        # Test matmul
        shape_info = {"A": (100, 50), "B": (50, 75)}
        self.timing.start("matmul_test", operation_type="matmul", shape_info=shape_info)
        self.timing.stop("matmul_test")

        section = self.timing._sections["matmul_test"]
        expected_flops = 2 * 100 * 75 * 50  # 2 * M * N * K
        self.assertEqual(section.flops, expected_flops)

    def test_bandwidth_calculation(self):
        """Test memory bandwidth calculations."""
        # Create section with memory operations
        self.timing.start("bandwidth_test", read=1024 * 1024 * 100, written=1024 * 1024 * 50)  # 100MB read, 50MB write
        time.sleep(0.1)  # 100ms duration
        self.timing.stop("bandwidth_test")

        section = self.timing._sections["bandwidth_test"]
        bandwidth_metrics = self.timing._calculate_bandwidth_metrics(section)

        self.assertIn("memory_bandwidth_gbps", bandwidth_metrics)
        self.assertIn("read_bandwidth_gbps", bandwidth_metrics)
        self.assertIn("write_bandwidth_gbps", bandwidth_metrics)

        # Check that bandwidth is reasonable (should be around 1.5 GB/s for 150MB in 0.1s)
        self.assertGreater(bandwidth_metrics["memory_bandwidth_gbps"], 1.0)
        self.assertLess(bandwidth_metrics["memory_bandwidth_gbps"], 2.0)

    def test_flops_calculation(self):
        """Test FLOPS calculations."""
        # Create section with FLOPS
        flops_count = 1000000000  # 1 billion FLOPS
        self.timing.start("flops_test", flops=flops_count)
        time.sleep(0.1)  # 100ms duration
        self.timing.stop("flops_test")

        section = self.timing._sections["flops_test"]
        flops_metrics = self.timing._calculate_flops_metrics(section)

        self.assertIn("gflops", flops_metrics)
        self.assertIn("gflops_per_second", flops_metrics)

        self.assertAlmostEqual(flops_metrics["gflops"], 1.0, places=1)
        self.assertAlmostEqual(flops_metrics["gflops_per_second"], 10.0, places=1)

    def test_error_handling(self):
        """Test error handling in timing operations."""
        # Test mismatched start/stop
        self.timing.start("section1")
        result = self.timing.stop("section2")  # Wrong name
        self.assertFalse(result)

        # Test stopping without starting
        result = self.timing.stop("nonexistent")
        self.assertFalse(result)

        # Error count should be tracked
        self.assertGreater(self.timing._error_count, 0)

    def test_null_safe_pattern(self):
        """Test null-safe usage pattern."""
        # Test with None timing instance
        timing = None

        # These should not raise exceptions
        result = timing and timing.start("test")
        self.assertFalse(result)

        result = timing and timing.stop("test")
        self.assertFalse(result)

        # Test with context manager
        with timing.section("test") if timing else self._null_context():
            pass  # Should not raise exception

    def _null_context(self):
        """Helper for null context manager test."""

        class NullContext:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return NullContext()

    def test_logging_output(self):
        """Test log output formatting."""
        # Create some test data
        self.timing.start("main_section", read=1000, written=500, flops=1000000)
        time.sleep(0.01)

        self.timing.start("nested_section", flops=500000)
        time.sleep(0.005)
        self.timing.stop("nested_section")

        self.timing.stop("main_section")

        # Test logging
        self.timing.log(self.logger, include_percentages=True)

        # Verify logger was called
        self.assertTrue(self.logger.info.called)

        # Check that report includes expected information
        log_calls = [call[0][0] for call in self.logger.info.call_args_list]
        log_text = " ".join(log_calls)

        self.assertIn("PerformanceTiming Report", log_text)
        self.assertIn("main_section", log_text)
        self.assertIn("nested_section", log_text)
        self.assertIn("Duration:", log_text)
        self.assertIn("FLOPS:", log_text)

    def test_get_timing_data(self):
        """Test structured timing data retrieval."""
        self.timing.start("test_section", read=1000, flops=2000)
        time.sleep(0.01)
        self.timing.stop("test_section")

        data = self.timing.get_timing_data()

        # Verify structure
        self.assertIn("module_name", data)
        self.assertIn("sections", data)
        self.assertIn("summary", data)

        self.assertEqual(data["module_name"], "TestModule")
        self.assertIn("test_section", data["sections"])

        section_data = data["sections"]["test_section"]
        self.assertEqual(section_data["read_bytes"], 1000)
        self.assertEqual(section_data["flops"], 2000)
        self.assertGreater(section_data["duration"], 0)

    def test_csv_export(self):
        """Test CSV export functionality."""
        # Create test data
        self.timing.start("export_test", read=1000, written=500, flops=1000000)
        time.sleep(0.01)
        self.timing.stop("export_test")

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_filename = f.name

        try:
            result = self.timing.export_csv(temp_filename)
            self.assertTrue(result)

            # Verify CSV content
            with open(temp_filename) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            row = rows[0]

            self.assertEqual(row["section_name"], "export_test")
            self.assertEqual(int(row["read_bytes"]), 1000)
            self.assertEqual(int(row["written_bytes"]), 500)
            self.assertEqual(int(row["flops"]), 1000000)
            self.assertGreater(float(row["duration"]), 0)

        finally:
            os.unlink(temp_filename)

    def test_reset_functionality(self):
        """Test reset functionality."""
        # Create some timing data
        self.timing.start("test_section")
        time.sleep(0.01)
        self.timing.stop("test_section")

        self.assertEqual(len(self.timing._sections), 1)

        # Reset
        self.timing.reset()

        # Verify everything is cleared
        self.assertEqual(len(self.timing._sections), 0)
        self.assertEqual(len(self.timing._section_stack), 0)
        self.assertIsNone(self.timing._current_section)
        self.assertEqual(self.timing._error_count, 0)
        self.assertIsNone(self.timing._last_error)

    def test_get_stats(self):
        """Test statistics summary."""
        # Create test data
        self.timing.start("stats_test", flops=1000000)
        time.sleep(0.01)
        self.timing.stop("stats_test")

        stats = self.timing.get_stats()

        self.assertEqual(stats["module_name"], "TestModule")
        self.assertEqual(stats["section_count"], 1)
        self.assertGreater(stats["total_duration"], 0)
        self.assertEqual(stats["total_flops"], 1000000)
        self.assertEqual(stats["error_count"], 0)
        self.assertIn("gpu_available", stats)
        self.assertIn("gpu_timing_enabled", stats)

    def test_sorting_options(self):
        """Test different sorting options for log output."""
        # Create multiple sections with different characteristics
        sections_data = [
            ("fast_section", 0.001, 100),
            ("slow_section", 0.1, 1000),
            ("high_flops_section", 0.05, 10000),
        ]

        for name, sleep_time, flops in sections_data:
            self.timing.start(name, flops=flops)
            time.sleep(sleep_time)
            self.timing.stop(name)

        # Test sorting by time
        self.timing.log(self.logger, sort_by="time")
        self.assertTrue(self.logger.info.called)

        # Test sorting by flops
        self.logger.reset_mock()
        self.timing.log(self.logger, sort_by="flops")
        self.assertTrue(self.logger.info.called)

        # Test sorting by name (default)
        self.logger.reset_mock()
        self.timing.log(self.logger, sort_by="name")
        self.assertTrue(self.logger.info.called)

    def test_deeply_nested_sections(self):
        """Test deeply nested timing sections."""
        depth = 5
        section_names = [f"level_{i}" for i in range(depth)]

        # Start nested sections
        for name in section_names:
            self.timing.start(name)
            time.sleep(0.001)

        # Stop nested sections in reverse order
        for name in reversed(section_names):
            self.timing.stop(name)

        # Verify nesting structure
        for i, name in enumerate(section_names):
            section = self.timing._sections[name]
            self.assertEqual(section.depth, i)

            if i > 0:
                parent_name = section_names[i - 1]
                parent = self.timing._sections[parent_name]
                self.assertEqual(section.parent, parent)
                self.assertIn(section, parent.children)

    @patch("src.utils.performance_timing.GPU_AVAILABLE", True)
    @patch("src.utils.performance_timing.cp")
    def test_gpu_timing_mocked(self, mock_cp):
        """Test GPU timing functionality with mocked CuPy."""
        # Mock GPU events
        mock_start_event = Mock()
        mock_end_event = Mock()
        mock_cp.cuda.Event.side_effect = [mock_start_event, mock_end_event]
        mock_cp.cuda.get_elapsed_time.return_value = 50.0  # 50ms

        timing = PerformanceTiming("GPUTest", enable_gpu_timing=True)

        timing.start("gpu_section", use_gpu_events=True)
        time.sleep(0.01)
        timing.stop("gpu_section")

        # Verify GPU events were created and used
        self.assertEqual(mock_cp.cuda.Event.call_count, 2)
        # Note: GPU duration would be set if CuPy was actually available


class TestTimingSection(unittest.TestCase):
    """Test cases for TimingSection data structure."""

    def test_timing_section_creation(self):
        """Test TimingSection creation and initialization."""
        section = TimingSection(name="test_section")

        self.assertEqual(section.name, "test_section")
        self.assertIsNone(section.start_time)
        self.assertIsNone(section.end_time)
        self.assertEqual(section.duration, 0.0)
        self.assertEqual(section.read_bytes, 0)
        self.assertEqual(section.written_bytes, 0)
        self.assertEqual(section.flops, 0)
        self.assertEqual(section.occurrence_count, 0)
        self.assertEqual(len(section.children), 0)
        self.assertEqual(len(section.memory_transfers), 0)


class TestTimingSectionContext(unittest.TestCase):
    """Test cases for TimingSectionContext."""

    def setUp(self):
        """Set up test fixtures."""
        self.timing = PerformanceTiming("ContextTest")

    def test_context_creation(self):
        """Test context manager creation."""
        context = TimingSectionContext(self.timing, "test_context")

        self.assertEqual(context.timing, self.timing)
        self.assertEqual(context.section_name, "test_context")

    def test_context_with_none_timing(self):
        """Test context manager with None timing instance."""
        context = TimingSectionContext(None, "test_context")

        # Should not raise exception
        with context:
            pass


if __name__ == "__main__":
    unittest.main()
