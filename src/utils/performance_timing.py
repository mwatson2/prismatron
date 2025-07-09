"""
PerformanceTiming - Lightweight performance measurement and FLOPS analysis.

This module provides a comprehensive timing and performance analysis framework
specifically designed for GPU-accelerated scientific computing applications.
Key features include nested timing sections, GPU event timing, memory bandwidth
calculations, and FLOPS analysis.
"""

import contextlib
import csv
import logging
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# GPU support - conditionally imported
try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@dataclass
class TimingSection:
    """Data structure for a single timing section."""

    name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: float = 0.0

    # Performance metrics
    read_bytes: int = 0
    written_bytes: int = 0
    flops: int = 0

    # GPU timing
    use_gpu_events: bool = False
    gpu_start_event = None
    gpu_end_event = None
    gpu_duration: Optional[float] = None

    # Memory transfer tracking
    memory_transfers: List[Dict[str, Any]] = field(default_factory=list)

    # Nested sections
    parent: Optional["TimingSection"] = None
    children: List["TimingSection"] = field(default_factory=list)
    depth: int = 0

    # Occurrence tracking
    occurrence_count: int = 0
    first_occurrence_data: Optional[Dict[str, Any]] = None
    occurrence_durations: List[float] = field(default_factory=list)
    occurrence_metrics: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OperationShapeInfo:
    """Shape information for automatic FLOPS calculation."""

    operation_type: str
    shapes: Dict[str, Tuple[int, ...]]
    additional_params: Dict[str, Any] = field(default_factory=dict)


class TimingSectionContext:
    """Context manager for timing sections."""

    def __init__(self, timing_instance: "PerformanceTiming", section_name: str, **kwargs):
        self.timing = timing_instance
        self.section_name = section_name
        self.kwargs = kwargs

    def __enter__(self):
        if self.timing:
            self.timing.start(self.section_name, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timing:
            self.timing.stop(self.section_name)

    def add_memory(self, read: int = 0, written: int = 0):
        """Add additional memory information during the operation."""
        if self.timing and self.timing._current_section:
            self.timing._current_section.read_bytes += read
            self.timing._current_section.written_bytes += written

    def record_memory_transfer(self, size_mb: float, direction: str):
        """Record a memory transfer operation."""
        if self.timing:
            self.timing.record_memory_transfer(size_mb, direction)


class PerformanceTiming:
    """
    Lightweight performance measurement and FLOPS analysis framework.

    Provides comprehensive timing capabilities including nested sections,
    GPU event timing, memory bandwidth calculations, and FLOPS analysis.
    Designed to be lightweight with minimal overhead when disabled.
    """

    def __init__(self, module_name: str, enable: bool = True, enable_gpu_timing: bool = True):
        """
        Initialize performance timing for a module.

        Args:
            module_name: Name of the module being timed
            enable_gpu_timing: Enable GPU event timing when available
        """
        self.module_name = module_name
        self.enabled = enable
        self.enable_gpu_timing = enable_gpu_timing and GPU_AVAILABLE

        # Timing state
        self._sections: Dict[str, TimingSection] = {}
        self._section_stack: List[TimingSection] = []
        self._current_section: Optional[TimingSection] = None
        self._start_time = time.time()

        # Error handling
        self._error_count = 0
        self._last_error: Optional[str] = None

        # Performance tracking
        self._total_gpu_memory_transfers = 0
        self._total_cpu_gpu_transfers = 0

    def _safe_execute(self, func, *args, **kwargs):
        """Safely execute a timing function with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            # Log error but don't crash the application
            return None

    def _get_or_create_section(self, name: str) -> TimingSection:
        """Get existing section or create new one."""
        if name not in self._sections:
            self._sections[name] = TimingSection(
                name=name, depth=len(self._section_stack), parent=self._current_section
            )
            if self._current_section:
                self._current_section.children.append(self._sections[name])
        return self._sections[name]

    def _gpu_sync(self):
        """Perform GPU synchronization if available."""
        if self.enable_gpu_timing and cp is not None:
            with contextlib.suppress(Exception):
                cp.cuda.runtime.deviceSynchronize()

    def _create_gpu_events(self):
        """Create GPU timing events if available."""
        if self.enable_gpu_timing and cp is not None:
            try:
                return cp.cuda.Event(), cp.cuda.Event()
            except Exception:
                return None, None  # nosec B110 Graceful fallback if GPU operations fail
        return None, None

    def _calculate_automatic_flops(self, operation_type: str, shape_info: Dict[str, Tuple[int, ...]]) -> int:
        """Calculate FLOPS automatically based on operation type and shapes."""
        try:
            if operation_type == "einsum":
                # Simplified einsum FLOPS calculation
                # This would need more sophisticated parsing for complex einsum equations
                total_elements = 1
                for shape in shape_info.values():
                    total_elements *= max(shape) if shape else 1
                return total_elements * 2  # Multiply + accumulate

            elif operation_type == "matmul":
                if "A" in shape_info and "B" in shape_info:
                    A_shape = shape_info["A"]
                    B_shape = shape_info["B"]
                    if len(A_shape) >= 2 and len(B_shape) >= 2:
                        # Standard matrix multiplication FLOPS: 2 * M * N * K
                        M, K = A_shape[-2], A_shape[-1]
                        N = B_shape[-1]
                        return 2 * M * N * K

            elif operation_type == "elementwise":
                total_elements = 1
                for shape in shape_info.values():
                    elements = 1
                    for dim in shape:
                        elements *= dim
                    total_elements = max(total_elements, elements)
                return total_elements  # One operation per element

            elif operation_type == "sparse_matmul":
                # Would need non-zero count from additional_params
                nnz = shape_info.get("nnz", 0)
                return nnz * 2  # Multiply + accumulate per non-zero

        except Exception:
            # Graceful fallback for FLOPS calculation when shape info is incomplete/invalid
            pass  # nosec B110

        return 0  # Fallback for unknown operations

    def start(
        self,
        section_name: str,
        read: int = 0,
        written: int = 0,
        flops: int = 0,
        use_gpu_events: bool = False,
        operation_type: Optional[str] = None,
        shape_info: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> bool:
        """
        Start timing a section.

        Args:
            section_name: Name of the section to time
            read: Bytes read (optional, can be provided here or in stop())
            written: Bytes written (optional, can be provided here or in stop())
            flops: FLOPS count (optional, can be provided here or in stop())
            use_gpu_events: Use GPU events for more accurate timing
            operation_type: Type of operation for automatic FLOPS calculation
            shape_info: Shape information for automatic FLOPS calculation

        Returns:
            True if started successfully, False otherwise
        """

        def _start_impl():
            # GPU synchronization if requested
            if use_gpu_events:
                self._gpu_sync()

            section = self._get_or_create_section(section_name)
            section.use_gpu_events = use_gpu_events
            section.read_bytes += read
            section.written_bytes += written
            section.flops += flops

            # Automatic FLOPS calculation
            if operation_type and shape_info and flops == 0:
                auto_flops = self._calculate_automatic_flops(operation_type, shape_info)
                section.flops += auto_flops

            # Set up GPU timing if requested
            if use_gpu_events:
                (
                    section.gpu_start_event,
                    section.gpu_end_event,
                ) = self._create_gpu_events()
                if section.gpu_start_event:
                    with contextlib.suppress(Exception):
                        section.gpu_start_event.record()

            # Start CPU timing
            section.start_time = time.time()

            # Update stack state
            self._section_stack.append(section)
            self._current_section = section

            return True

        return (self._safe_execute(_start_impl) is not False) if self.enabled else True

    def stop(self, section_name: str, read: int = 0, written: int = 0, flops: int = 0) -> bool:
        """
        Stop timing a section.

        Args:
            section_name: Name of the section to stop timing
            read: Additional bytes read
            written: Additional bytes written
            flops: Additional FLOPS count

        Returns:
            True if stopped successfully, False otherwise
        """

        def _stop_impl():
            if not self._section_stack:
                self._error_count += 1
                self._last_error = f"No sections started for stop: {section_name}"
                return False

            section = self._section_stack[-1]
            if section.name != section_name:
                # Handle nested section mismatch
                self._error_count += 1
                self._last_error = f"Section mismatch: expected {section.name}, got {section_name}"
                return False

            # Stop CPU timing
            section.end_time = time.time()
            section.duration = section.end_time - (section.start_time or section.end_time)

            # Add additional metrics
            section.read_bytes += read
            section.written_bytes += written
            section.flops += flops

            # Stop GPU timing if used
            if section.use_gpu_events and section.gpu_end_event:
                try:
                    section.gpu_end_event.record()
                    cp.cuda.runtime.deviceSynchronize()
                    section.gpu_duration = (
                        cp.cuda.get_elapsed_time(section.gpu_start_event, section.gpu_end_event) / 1000.0
                    )  # Convert ms to seconds
                except Exception:
                    section.gpu_duration = None  # nosec B110 Graceful fallback if GPU operations fail

            # Update occurrence tracking
            section.occurrence_count += 1
            section.occurrence_durations.append(section.duration)

            # Store first occurrence data separately
            if section.occurrence_count == 1:
                section.first_occurrence_data = {
                    "duration": section.duration,
                    "read_bytes": section.read_bytes,
                    "written_bytes": section.written_bytes,
                    "flops": section.flops,
                    "gpu_duration": section.gpu_duration,
                }

            # Store occurrence metrics
            section.occurrence_metrics.append(
                {
                    "duration": section.duration,
                    "read_bytes": section.read_bytes,
                    "written_bytes": section.written_bytes,
                    "flops": section.flops,
                    "gpu_duration": section.gpu_duration,
                }
            )

            # Update stack state
            self._section_stack.pop()
            self._current_section = self._section_stack[-1] if self._section_stack else None

            return True

        return (self._safe_execute(_stop_impl) is not False) if self.enabled else True

    def record_memory_transfer(self, size_mb: float, direction: str):
        """
        Record a memory transfer operation.

        Args:
            size_mb: Size of transfer in megabytes
            direction: Transfer direction ("cpu_to_gpu", "gpu_to_cpu", "gpu_to_gpu")
        """

        def _record_impl():
            transfer_info = {
                "size_mb": size_mb,
                "direction": direction,
                "timestamp": time.time(),
            }

            if self._current_section:
                self._current_section.memory_transfers.append(transfer_info)

            # Update global counters
            if "gpu" in direction:
                self._total_gpu_memory_transfers += 1
            if "cpu" in direction and "gpu" in direction:
                self._total_cpu_gpu_transfers += 1

        return self._safe_execute(_record_impl)

    @contextmanager
    def section(self, section_name: str, **kwargs):
        """
        Context manager for timing sections.

        Args:
            section_name: Name of the section
            **kwargs: Arguments passed to start() method

        Returns:
            TimingSectionContext for additional operations
        """
        if self.enabled:
            context = TimingSectionContext(self, section_name, **kwargs)
            try:
                yield context.__enter__()
            finally:
                context.__exit__(None, None, None)

    def _calculate_bandwidth_metrics(self, section: TimingSection) -> Dict[str, float]:
        """Calculate memory bandwidth metrics for a section."""
        metrics = {}

        if section.duration > 0:
            total_bytes = section.read_bytes + section.written_bytes
            if total_bytes > 0:
                # Computational memory bandwidth (GB/s)
                metrics["memory_bandwidth_gbps"] = (total_bytes / section.duration) / (1024**3)

            # Individual read/write bandwidths
            if section.read_bytes > 0:
                metrics["read_bandwidth_gbps"] = (section.read_bytes / section.duration) / (1024**3)
            if section.written_bytes > 0:
                metrics["write_bandwidth_gbps"] = (section.written_bytes / section.duration) / (1024**3)

        # Memory transfer bandwidth
        total_transfer_mb = sum(t["size_mb"] for t in section.memory_transfers)
        if total_transfer_mb > 0 and section.duration > 0:
            metrics["transfer_bandwidth_gbps"] = (total_transfer_mb / section.duration) / 1024

        return metrics

    def _calculate_flops_metrics(self, section: TimingSection) -> Dict[str, float]:
        """Calculate FLOPS metrics for a section."""
        metrics = {}

        if section.flops > 0 and section.duration > 0:
            metrics["gflops"] = section.flops / 1e9
            metrics["gflops_per_second"] = (section.flops / section.duration) / 1e9

        return metrics

    def _format_section_report(
        self,
        section: TimingSection,
        include_percentages: bool = False,
        total_time: float = 0.0,
    ) -> List[str]:
        """Format a detailed report for a timing section."""
        lines = []
        indent = "  " * section.depth

        # Calculate metrics
        bandwidth_metrics = self._calculate_bandwidth_metrics(section)
        flops_metrics = self._calculate_flops_metrics(section)

        # Section header
        header = f"{indent}{section.name}"
        if include_percentages and total_time > 0:
            percentage = (section.duration / total_time) * 100
            header += f" ({percentage:.1f}%)"
        lines.append(header)

        # Basic timing info
        lines.append(f"{indent}  Duration: {section.duration:.3f}s")
        lines.append(f"{indent}  Occurrences: {section.occurrence_count}")

        # GPU timing if available
        if section.gpu_duration is not None:
            lines.append(f"{indent}  GPU Duration: {section.gpu_duration:.3f}s")

        # Memory bandwidth
        if bandwidth_metrics:
            lines.append(f"{indent}  Memory Bandwidth:")
            for key, value in bandwidth_metrics.items():
                lines.append(f"{indent}    {key}: {value:.2f}")

        # FLOPS metrics
        if flops_metrics:
            lines.append(f"{indent}  FLOPS:")
            for key, value in flops_metrics.items():
                lines.append(f"{indent}    {key}: {value:.2f}")

        # Occurrence statistics for repeated sections
        if section.occurrence_count > 1:
            durations = section.occurrence_durations
            if durations:
                avg_duration = statistics.mean(durations)
                std_duration = statistics.stdev(durations) if len(durations) > 1 else 0

                lines.append(f"{indent}  First occurrence: {durations[0]:.3f}s")
                lines.append(f"{indent}  Average duration: {avg_duration:.3f}s")
                lines.append(f"{indent}  Std deviation: {std_duration:.3f}s")

        # Memory transfer summary
        if section.memory_transfers:
            total_mb = sum(t["size_mb"] for t in section.memory_transfers)
            lines.append(f"{indent}  Memory transfers: {len(section.memory_transfers)} ({total_mb:.1f}MB)")

        return lines

    def log(
        self,
        logger: logging.Logger,
        include_percentages: bool = False,
        sort_by: str = "name",
    ) -> None:
        """
        Log comprehensive timing report.

        Args:
            logger: Logger instance to use
            include_percentages: Include percentage of total time
            sort_by: Sort sections by "name", "time", or "flops"
        """

        def _log_impl():
            if not self._sections:
                logger.info(f"PerformanceTiming[{self.module_name}]: No sections timed")
                return

            total_time = time.time() - self._start_time

            # Sort sections
            sections = list(self._sections.values())
            if sort_by == "time":
                sections.sort(key=lambda s: s.duration, reverse=True)
            elif sort_by == "flops":
                sections.sort(key=lambda s: s.flops, reverse=True)
            else:  # sort by name
                sections.sort(key=lambda s: s.name)

            # Generate report
            logger.info(f"PerformanceTiming Report [{self.module_name}]")
            logger.info(f"Total elapsed time: {total_time:.3f}s")

            if self._error_count > 0:
                logger.warning(f"Timing errors encountered: {self._error_count}")

            # Report each section
            for section in sections:
                if section.depth == 0:  # Only report top-level sections, children are handled recursively
                    report_lines = self._format_section_report(section, include_percentages, total_time)
                    for line in report_lines:
                        logger.info(line)

                    # Report children recursively
                    self._log_children(logger, section, include_percentages, total_time)

        self._safe_execute(_log_impl)

    def _log_children(
        self,
        logger: logging.Logger,
        parent: TimingSection,
        include_percentages: bool,
        total_time: float,
    ):
        """Recursively log child sections."""
        for child in parent.children:
            report_lines = self._format_section_report(child, include_percentages, total_time)
            for line in report_lines:
                logger.info(line)
            self._log_children(logger, child, include_percentages, total_time)

    def get_timing_data(self) -> Dict[str, Any]:
        """
        Get structured timing data for analysis.

        Returns:
            Dictionary containing all timing data
        """

        def _get_data_impl():
            data = {
                "module_name": self.module_name,
                "total_elapsed_time": time.time() - self._start_time,
                "error_count": self._error_count,
                "last_error": self._last_error,
                "sections": {},
                "summary": {
                    "total_sections": len(self._sections),
                    "total_gpu_transfers": self._total_gpu_memory_transfers,
                    "total_cpu_gpu_transfers": self._total_cpu_gpu_transfers,
                },
            }

            for name, section in self._sections.items():
                data["sections"][name] = {
                    "name": section.name,
                    "duration": section.duration,
                    "occurrence_count": section.occurrence_count,
                    "read_bytes": section.read_bytes,
                    "written_bytes": section.written_bytes,
                    "flops": section.flops,
                    "gpu_duration": section.gpu_duration,
                    "depth": section.depth,
                    "memory_transfers": section.memory_transfers,
                    "occurrence_durations": section.occurrence_durations,
                    "bandwidth_metrics": self._calculate_bandwidth_metrics(section),
                    "flops_metrics": self._calculate_flops_metrics(section),
                }

            return data

        return self._safe_execute(_get_data_impl) or {}

    def export_csv(self, filename: str) -> bool:
        """
        Export timing data to CSV file.

        Args:
            filename: Output CSV filename

        Returns:
            True if exported successfully, False otherwise
        """

        def _export_impl():
            data = self.get_timing_data()

            with open(filename, "w", newline="") as csvfile:
                fieldnames = [
                    "section_name",
                    "duration",
                    "occurrence_count",
                    "read_bytes",
                    "written_bytes",
                    "flops",
                    "memory_bandwidth_gbps",
                    "gflops_per_second",
                    "gpu_duration",
                    "depth",
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for section_data in data["sections"].values():
                    row = {
                        "section_name": section_data["name"],
                        "duration": section_data["duration"],
                        "occurrence_count": section_data["occurrence_count"],
                        "read_bytes": section_data["read_bytes"],
                        "written_bytes": section_data["written_bytes"],
                        "flops": section_data["flops"],
                        "gpu_duration": section_data["gpu_duration"],
                        "depth": section_data["depth"],
                    }

                    # Add bandwidth and FLOPS metrics
                    bandwidth_metrics = section_data.get("bandwidth_metrics", {})
                    flops_metrics = section_data.get("flops_metrics", {})

                    row["memory_bandwidth_gbps"] = bandwidth_metrics.get("memory_bandwidth_gbps", 0)
                    row["gflops_per_second"] = flops_metrics.get("gflops_per_second", 0)

                    writer.writerow(row)

            return True

        return self._safe_execute(_export_impl) is not False

    def reset(self):
        """Reset all timing data."""

        def _reset_impl():
            self._sections.clear()
            self._section_stack.clear()
            self._current_section = None
            self._start_time = time.time()
            self._error_count = 0
            self._last_error = None
            self._total_gpu_memory_transfers = 0
            self._total_cpu_gpu_transfers = 0

        self._safe_execute(_reset_impl)

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_time = sum(s.duration for s in self._sections.values())
        total_flops = sum(s.flops for s in self._sections.values())

        return {
            "module_name": self.module_name,
            "section_count": len(self._sections),
            "total_duration": total_time,
            "total_flops": total_flops,
            "error_count": self._error_count,
            "gpu_available": GPU_AVAILABLE,
            "gpu_timing_enabled": self.enable_gpu_timing,
        }
