"""
Unit tests for the Video Converter.

Tests video conversion job management, FFmpeg integration,
progress tracking, and file handling.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.web.video_converter import (
    ConversionJob,
    ConversionManager,
    ConversionStatus,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for testing."""
    uploads_dir = tmp_path / "uploads"
    temp_conversions = tmp_path / "temp_conversions"
    uploads_dir.mkdir()
    temp_conversions.mkdir()
    return {"uploads": uploads_dir, "temp_conversions": temp_conversions}


@pytest.fixture
def sample_config():
    """Create sample conversion configuration."""
    return {
        "output_format": {
            "width": 800,
            "height": 480,
            "codec": "libx264",
            "profile": "high",
            "level": "3.1",
            "pixel_format": "yuv420p",
            "preset": "fast",
            "crf": 23,
        }
    }


@pytest.fixture
def conversion_manager(sample_config, temp_dirs):
    """Create a ConversionManager for testing."""
    with patch("src.paths.TEMP_CONVERSIONS_DIR", temp_dirs["temp_conversions"]), patch(
        "src.paths.UPLOADS_DIR", temp_dirs["uploads"]
    ):
        manager = ConversionManager(sample_config)
        manager.temp_dir = temp_dirs["temp_conversions"]
        manager.uploads_dir = temp_dirs["uploads"]
        yield manager
        # Cleanup
        manager.stop()


@pytest.fixture
def sample_video_file(temp_dirs):
    """Create a sample video file for testing."""
    video_path = temp_dirs["temp_conversions"] / "test_video_upload.mp4"
    # Create a minimal file (not a real video, just for path testing)
    video_path.write_bytes(b"fake video content")
    return video_path


@pytest.fixture
def sample_video_info():
    """Sample video analysis info."""
    return {
        "width": 1920,
        "height": 1080,
        "duration": 30.0,
        "fps": 30.0,
        "codec": "h264",
        "pixel_format": "yuv420p",
    }


# =============================================================================
# ConversionStatus Tests
# =============================================================================


class TestConversionStatus:
    """Test ConversionStatus enum."""

    def test_status_values(self):
        """Test that all status values are defined."""
        assert ConversionStatus.QUEUED.value == "queued"
        assert ConversionStatus.PROCESSING.value == "processing"
        assert ConversionStatus.VALIDATING.value == "validating"
        assert ConversionStatus.COMPLETED.value == "completed"
        assert ConversionStatus.FAILED.value == "failed"
        assert ConversionStatus.CANCELLED.value == "cancelled"

    def test_status_is_string_enum(self):
        """Test that status values can be used as strings."""
        status = ConversionStatus.QUEUED
        assert str(status.value) == "queued"


# =============================================================================
# ConversionJob Tests
# =============================================================================


class TestConversionJob:
    """Test ConversionJob dataclass."""

    def test_job_creation(self, temp_dirs):
        """Test creating a conversion job."""
        job = ConversionJob(
            id="test-123",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        assert job.id == "test-123"
        assert job.status == ConversionStatus.QUEUED
        assert job.progress == 0.0
        assert job.error_message is None

    def test_job_to_dict(self, temp_dirs):
        """Test job serialization to dictionary."""
        job = ConversionJob(
            id="test-123",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
            status=ConversionStatus.PROCESSING,
            progress=50.0,
        )

        data = job.to_dict()

        assert data["id"] == "test-123"
        assert data["original_name"] == "output.mp4"
        assert data["status"] == "processing"
        assert data["progress"] == 50.0
        assert "created_at" in data

    def test_job_to_dict_with_dates(self, temp_dirs):
        """Test job serialization includes dates correctly."""
        now = datetime.now()
        job = ConversionJob(
            id="test-123",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
            created_at=now,
            started_at=now,
            completed_at=now,
        )

        data = job.to_dict()

        assert data["created_at"] == now.isoformat()
        assert data["started_at"] == now.isoformat()
        assert data["completed_at"] == now.isoformat()


# =============================================================================
# ConversionManager Initialization Tests
# =============================================================================


class TestConversionManagerInit:
    """Test ConversionManager initialization."""

    def test_manager_initialization(self, sample_config, temp_dirs):
        """Test manager initializes correctly."""
        with patch("src.paths.TEMP_CONVERSIONS_DIR", temp_dirs["temp_conversions"]), patch(
            "src.paths.UPLOADS_DIR", temp_dirs["uploads"]
        ):
            manager = ConversionManager(sample_config)

            assert manager.config == sample_config
            assert manager.conversion_queue == []
            assert manager.completed_jobs == []
            assert manager.current_job is None
            assert manager.worker_thread is None

            manager.stop()

    def test_manager_start_creates_worker_thread(self, conversion_manager):
        """Test that start() creates worker thread."""
        conversion_manager.start()

        assert conversion_manager.worker_thread is not None
        assert conversion_manager.worker_thread.is_alive()


# =============================================================================
# Job Queue Management Tests
# =============================================================================


class TestJobQueueManagement:
    """Test job queue management."""

    def test_queue_conversion(self, sample_config, temp_dirs):
        """Test queuing a conversion job."""
        # Create manager without starting worker
        with patch("src.paths.TEMP_CONVERSIONS_DIR", temp_dirs["temp_conversions"]), patch(
            "src.paths.UPLOADS_DIR", temp_dirs["uploads"]
        ):
            manager = ConversionManager(sample_config)
            manager.temp_dir = temp_dirs["temp_conversions"]
            manager.uploads_dir = temp_dirs["uploads"]

            # Create a sample video file
            sample_video_file = temp_dirs["temp_conversions"] / "test_video_upload.mp4"
            sample_video_file.write_bytes(b"fake video content")

            # Don't call start() to avoid processing
            # Queue the job manually without starting the worker
            job = manager.queue_conversion(sample_video_file, "test_output.mp4")

            # Stop worker immediately to prevent processing
            manager.stop()

            assert job is not None
            assert job.original_name == "test_output.mp4"

    def test_get_job_by_id(self, conversion_manager, temp_dirs):
        """Test retrieving job by ID."""
        # Create a job directly in the queue
        job = ConversionJob(
            id="test-123",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )
        conversion_manager.conversion_queue.append(job)

        retrieved = conversion_manager.get_job("test-123")

        assert retrieved is not None
        assert retrieved.id == "test-123"

    def test_get_nonexistent_job(self, conversion_manager):
        """Test retrieving non-existent job returns None."""
        result = conversion_manager.get_job("nonexistent-id")
        assert result is None

    def test_get_all_jobs(self, conversion_manager, temp_dirs):
        """Test getting all jobs."""
        # Add some jobs
        job1 = ConversionJob(
            id="job-1",
            input_path=temp_dirs["temp_conversions"] / "input1.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output1_converted.mp4",
            output_path=temp_dirs["uploads"] / "output1.mp4",
            original_name="output1.mp4",
        )
        job2 = ConversionJob(
            id="job-2",
            input_path=temp_dirs["temp_conversions"] / "input2.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output2_converted.mp4",
            output_path=temp_dirs["uploads"] / "output2.mp4",
            original_name="output2.mp4",
        )

        conversion_manager.conversion_queue.append(job1)
        conversion_manager.conversion_queue.append(job2)

        all_jobs = conversion_manager.get_all_jobs()

        assert len(all_jobs) == 2

    def test_cancel_queued_job(self, conversion_manager, temp_dirs):
        """Test cancelling a queued job."""
        job = ConversionJob(
            id="cancel-me",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )
        conversion_manager.conversion_queue.append(job)

        result = conversion_manager.cancel_job("cancel-me")

        assert result is True
        assert len(conversion_manager.conversion_queue) == 0

    def test_cancel_nonexistent_job(self, conversion_manager):
        """Test cancelling non-existent job returns False."""
        result = conversion_manager.cancel_job("nonexistent")
        assert result is False

    def test_cancel_current_job(self, conversion_manager, temp_dirs):
        """Test cancelling the current job."""
        job = ConversionJob(
            id="current-job",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
            status=ConversionStatus.PROCESSING,
        )
        conversion_manager.current_job = job

        result = conversion_manager.cancel_job("current-job")

        assert result is True
        assert job.status == ConversionStatus.CANCELLED


# =============================================================================
# Video Analysis Tests
# =============================================================================


class TestVideoAnalysis:
    """Test video analysis functionality."""

    def test_analyze_video_success(self, conversion_manager, temp_dirs):
        """Test successful video analysis."""
        ffprobe_output = json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "30/1",
                        "pix_fmt": "yuv420p",
                    }
                ],
                "format": {"duration": "30.0"},
            }
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=ffprobe_output,
                stderr="",
            )

            result = conversion_manager._analyze_video(temp_dirs["temp_conversions"] / "test.mp4")

            assert result is not None
            assert result["width"] == 1920
            assert result["height"] == 1080
            assert result["fps"] == 30.0
            assert result["codec"] == "h264"

    def test_analyze_video_failure(self, conversion_manager, temp_dirs):
        """Test video analysis failure handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Error analyzing video",
            )

            result = conversion_manager._analyze_video(temp_dirs["temp_conversions"] / "test.mp4")

            assert result is None

    def test_analyze_video_no_video_stream(self, conversion_manager, temp_dirs):
        """Test handling video with no video stream."""
        ffprobe_output = json.dumps({"streams": [{"codec_type": "audio", "codec_name": "aac"}], "format": {}})

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=ffprobe_output,
                stderr="",
            )

            result = conversion_manager._analyze_video(temp_dirs["temp_conversions"] / "test.mp4")

            assert result is None


# =============================================================================
# FPS Parsing Tests
# =============================================================================


class TestFPSParsing:
    """Test FPS string parsing."""

    def test_parse_fps_fraction(self, conversion_manager):
        """Test parsing FPS as fraction."""
        assert conversion_manager._parse_fps("30/1") == 30.0
        assert conversion_manager._parse_fps("60/1") == 60.0
        assert conversion_manager._parse_fps("24000/1001") == pytest.approx(23.976, rel=0.01)

    def test_parse_fps_float(self, conversion_manager):
        """Test parsing FPS as float string."""
        assert conversion_manager._parse_fps("30.0") == 30.0
        assert conversion_manager._parse_fps("29.97") == pytest.approx(29.97, rel=0.01)

    def test_parse_fps_invalid(self, conversion_manager):
        """Test parsing invalid FPS returns default."""
        assert conversion_manager._parse_fps("invalid") == 30.0
        assert conversion_manager._parse_fps("") == 30.0

    def test_parse_fps_zero_denominator(self, conversion_manager):
        """Test parsing FPS with zero denominator."""
        assert conversion_manager._parse_fps("30/0") == 30.0


# =============================================================================
# Crop Filter Tests
# =============================================================================


class TestCropFilter:
    """Test crop filter calculation."""

    def test_calculate_crop_filter_wider_input(self, conversion_manager):
        """Test crop filter for wider input video."""
        result = conversion_manager._calculate_crop_filter(1920, 1080, 800, 480)

        # Should crop width for 16:9 to 5:3 (800x480)
        assert "crop=" in result

    def test_calculate_crop_filter_taller_input(self, conversion_manager):
        """Test crop filter for taller input video."""
        result = conversion_manager._calculate_crop_filter(1080, 1920, 800, 480)

        # Should crop height
        assert "crop=" in result

    def test_calculate_crop_filter_same_aspect(self, conversion_manager):
        """Test crop filter for same aspect ratio."""
        # 5:3 aspect ratio (same as 800x480)
        result = conversion_manager._calculate_crop_filter(1000, 600, 800, 480)

        # Should have minimal or no cropping
        assert "crop=" in result


# =============================================================================
# Status Callback Tests
# =============================================================================


class TestStatusCallbacks:
    """Test status update callbacks."""

    def test_add_status_callback(self, conversion_manager):
        """Test adding a status callback."""
        callback = MagicMock()
        conversion_manager.add_status_callback(callback)

        assert callback in conversion_manager.status_callbacks

    def test_notify_status_update(self, conversion_manager, temp_dirs):
        """Test that status callbacks are notified."""
        callback = MagicMock()
        conversion_manager.add_status_callback(callback)

        job = ConversionJob(
            id="test-123",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        conversion_manager._notify_status_update(job)

        callback.assert_called_once()


# =============================================================================
# Worker Thread Tests
# =============================================================================


class TestWorkerThread:
    """Test worker thread functionality."""

    def test_start_and_stop(self, conversion_manager):
        """Test starting and stopping worker thread."""
        conversion_manager.start()
        assert conversion_manager.worker_thread is not None
        assert conversion_manager.worker_thread.is_alive()

        conversion_manager.stop()
        # Give time for thread to stop
        time.sleep(0.1)
        assert not conversion_manager.worker_thread.is_alive()

    def test_worker_processes_queue(self, conversion_manager, temp_dirs):
        """Test that worker processes jobs from queue."""
        # Create a mock job
        job = ConversionJob(
            id="worker-test",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        # Mock the processing to avoid actual conversion
        with patch.object(conversion_manager, "_process_job") as mock_process:
            conversion_manager.conversion_queue.append(job)
            conversion_manager.start()

            # Wait for processing
            time.sleep(0.5)

            # Job should have been processed
            mock_process.assert_called()


# =============================================================================
# Job Processing Tests
# =============================================================================


class TestJobProcessing:
    """Test job processing logic."""

    def test_process_job_success(self, conversion_manager, temp_dirs, sample_video_info):
        """Test successful job processing."""
        job = ConversionJob(
            id="process-test",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        with patch.object(conversion_manager, "_analyze_video", return_value=sample_video_info), patch.object(
            conversion_manager, "_convert_video", return_value=True
        ), patch.object(conversion_manager, "_validate_output", return_value=True), patch.object(
            conversion_manager, "_finalize_conversion", return_value=True
        ):
            conversion_manager._process_job(job)

            assert job.status == ConversionStatus.COMPLETED
            assert job.progress == 100.0

    def test_process_job_analysis_failure(self, conversion_manager, temp_dirs):
        """Test job processing when analysis fails."""
        job = ConversionJob(
            id="fail-test",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        with patch.object(conversion_manager, "_analyze_video", return_value=None), patch.object(
            conversion_manager, "_cleanup_temp_files"
        ):
            conversion_manager._process_job(job)

            assert job.status == ConversionStatus.FAILED
            assert "analyze" in job.error_message.lower()

    def test_process_job_conversion_failure(self, conversion_manager, temp_dirs, sample_video_info):
        """Test job processing when conversion fails."""
        job = ConversionJob(
            id="convert-fail",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        with patch.object(conversion_manager, "_analyze_video", return_value=sample_video_info), patch.object(
            conversion_manager, "_convert_video", return_value=False
        ), patch.object(conversion_manager, "_cleanup_temp_files"):
            conversion_manager._process_job(job)

            assert job.status == ConversionStatus.FAILED

    def test_process_job_cancelled_during_conversion(self, conversion_manager, temp_dirs, sample_video_info):
        """Test job processing when cancelled during conversion."""
        job = ConversionJob(
            id="cancel-during",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "output_converted.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        def cancel_during_convert(j, info):
            j.status = ConversionStatus.CANCELLED
            return True

        with patch.object(conversion_manager, "_analyze_video", return_value=sample_video_info), patch.object(
            conversion_manager, "_convert_video", side_effect=cancel_during_convert
        ), patch.object(conversion_manager, "_cleanup_temp_files"):
            conversion_manager._process_job(job)

            assert job.status == ConversionStatus.CANCELLED


# =============================================================================
# File Cleanup Tests
# =============================================================================


class TestFileCleanup:
    """Test temporary file cleanup."""

    def test_cleanup_temp_files(self, conversion_manager, temp_dirs):
        """Test cleanup of temporary files."""
        # Create temp files
        input_file = temp_dirs["temp_conversions"] / "input_upload.mp4"
        temp_file = temp_dirs["temp_conversions"] / "output_converted.mp4"
        input_file.write_bytes(b"input")
        temp_file.write_bytes(b"temp")

        job = ConversionJob(
            id="cleanup-test",
            input_path=input_file,
            temp_path=temp_file,
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        conversion_manager._cleanup_temp_files(job)

        # Files should be removed
        assert not input_file.exists()
        assert not temp_file.exists()

    def test_cleanup_handles_missing_files(self, conversion_manager, temp_dirs):
        """Test cleanup handles missing files gracefully."""
        job = ConversionJob(
            id="missing-test",
            input_path=temp_dirs["temp_conversions"] / "nonexistent.mp4",
            temp_path=temp_dirs["temp_conversions"] / "also_nonexistent.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        # Should not raise exception
        conversion_manager._cleanup_temp_files(job)


# =============================================================================
# Output Validation Tests
# =============================================================================


class TestOutputValidation:
    """Test output video validation."""

    def test_validate_output_success(self, conversion_manager, temp_dirs):
        """Test successful output validation."""
        # Create temp output file
        temp_file = temp_dirs["temp_conversions"] / "output_converted.mp4"
        temp_file.write_bytes(b"video content")

        job = ConversionJob(
            id="validate-test",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_file,
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        ffprobe_output = json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 800,
                        "height": 480,
                        "pix_fmt": "yuv420p",
                    }
                ],
                "format": {"duration": "30.0"},
            }
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=ffprobe_output,
                stderr="",
            )

            result = conversion_manager._validate_output(job)

            assert result is True

    def test_validate_output_missing_file(self, conversion_manager, temp_dirs):
        """Test validation fails for missing file."""
        job = ConversionJob(
            id="missing-validate",
            input_path=temp_dirs["temp_conversions"] / "input.mp4",
            temp_path=temp_dirs["temp_conversions"] / "nonexistent.mp4",
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        result = conversion_manager._validate_output(job)

        assert result is False


# =============================================================================
# Finalization Tests
# =============================================================================


class TestFinalization:
    """Test conversion finalization."""

    def test_finalize_conversion_success(self, conversion_manager, temp_dirs):
        """Test successful finalization."""
        # Create temp output file
        temp_file = temp_dirs["temp_conversions"] / "output_converted.mp4"
        temp_file.write_bytes(b"video content")

        input_file = temp_dirs["temp_conversions"] / "input_upload.mp4"
        input_file.write_bytes(b"input content")

        job = ConversionJob(
            id="finalize-test",
            input_path=input_file,
            temp_path=temp_file,
            output_path=temp_dirs["uploads"] / "output.mp4",
            original_name="output.mp4",
        )

        result = conversion_manager._finalize_conversion(job)

        assert result is True
        assert job.output_path.exists()
        assert not temp_file.exists()


# =============================================================================
# Completed Jobs Cleanup Tests
# =============================================================================


class TestCompletedJobsCleanup:
    """Test cleanup of old completed jobs."""

    def test_old_completed_jobs_removed(self, conversion_manager, temp_dirs):
        """Test that old completed jobs are cleaned up."""
        old_job = ConversionJob(
            id="old-job",
            input_path=temp_dirs["temp_conversions"] / "old.mp4",
            temp_path=temp_dirs["temp_conversions"] / "old_converted.mp4",
            output_path=temp_dirs["uploads"] / "old_output.mp4",
            original_name="old_output.mp4",
            status=ConversionStatus.COMPLETED,
            completed_at=datetime.now() - timedelta(minutes=1),  # Older than 30 seconds
        )

        conversion_manager.completed_jobs.append(old_job)

        # Getting all jobs should clean up old ones
        all_jobs = conversion_manager.get_all_jobs()

        # Old job should be removed
        assert old_job not in conversion_manager.completed_jobs

    def test_recent_completed_jobs_kept(self, conversion_manager, temp_dirs):
        """Test that recent completed jobs are kept."""
        recent_job = ConversionJob(
            id="recent-job",
            input_path=temp_dirs["temp_conversions"] / "recent.mp4",
            temp_path=temp_dirs["temp_conversions"] / "recent_converted.mp4",
            output_path=temp_dirs["uploads"] / "recent_output.mp4",
            original_name="recent_output.mp4",
            status=ConversionStatus.COMPLETED,
            completed_at=datetime.now(),  # Just now
        )

        conversion_manager.completed_jobs.append(recent_job)

        all_jobs = conversion_manager.get_all_jobs()

        # Recent job should still be there
        assert recent_job in conversion_manager.completed_jobs
