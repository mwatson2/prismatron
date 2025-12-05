"""
Video Conversion System for Prismatron.

Automatically converts uploaded videos to H.264/800x480/8-bit format
optimized for hardware decoding on Jetson platform.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConversionStatus(Enum):
    """Status of a video conversion job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ConversionJob:
    """Represents a video conversion job."""

    id: str  # UUID for tracking
    input_path: Path  # Original file path in uploads/
    temp_path: Path  # Temporary file path during conversion
    output_path: Path  # Final converted file path in uploads/
    original_name: str  # Original filename
    status: ConversionStatus = ConversionStatus.QUEUED
    progress: float = 0.0  # 0.0 to 100.0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert job to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "original_name": self.original_name,
            "status": self.status.value,
            "progress": self.progress,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_duration": self.estimated_duration,
        }


class ConversionManager:
    """Manages video conversion queue and processing."""

    def __init__(self, config: Dict):
        """Initialize conversion manager with configuration."""
        self.config = config
        self.conversion_queue: List[ConversionJob] = []
        self.completed_jobs: List[ConversionJob] = []  # Keep completed jobs for frontend
        self.current_job: Optional[ConversionJob] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.status_callbacks: List[Callable] = []
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()

        # Use centralized paths module for temp directory
        from src.paths import TEMP_CONVERSIONS_DIR, UPLOADS_DIR

        self.temp_dir = TEMP_CONVERSIONS_DIR
        self.uploads_dir = UPLOADS_DIR

        logger.info(f"ConversionManager initialized with temp directory: {self.temp_dir}")

    def start(self):
        """Start the conversion worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.shutdown_event.clear()
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Conversion worker thread started")

    def stop(self):
        """Stop the conversion worker thread."""
        self.shutdown_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=30)
        logger.info("Conversion worker thread stopped")

    def add_status_callback(self, callback: Callable):
        """Add callback for status updates."""
        self.status_callbacks.append(callback)

    def queue_conversion(self, input_path: Path, original_name: str) -> ConversionJob:
        """Queue a video file for conversion."""
        job_id = str(uuid.uuid4())

        # Input file is already in temp conversions dir with _upload suffix
        # Output will be temp_conversions/{base_name}_converted.mp4
        # Final destination will be uploads/{original_name}
        base_name = Path(original_name).stem
        temp_output = self.temp_dir / f"{base_name}_converted.mp4"
        final_output = self.uploads_dir / original_name

        job = ConversionJob(
            id=job_id,
            input_path=input_path,  # Already in temp_conversions with _upload suffix
            temp_path=temp_output,
            output_path=final_output,
            original_name=original_name,
        )

        # Input file is already in temp directory, no need to copy
        logger.info(f"Queued conversion: {input_path} -> {final_output}")

        with self.lock:
            self.conversion_queue.append(job)

        logger.info(f"Queued conversion job: {job_id} for {original_name}")
        self._notify_status_update(job)

        # Start worker if not running
        self.start()

        return job

    def get_job(self, job_id: str) -> Optional[ConversionJob]:
        """Get job by ID."""
        with self.lock:
            # Check current job
            if self.current_job and self.current_job.id == job_id:
                return self.current_job

            # Check queue
            for job in self.conversion_queue:
                if job.id == job_id:
                    return job

        return None

    def get_all_jobs(self) -> List[ConversionJob]:
        """Get all jobs (current + queued + recently completed)."""
        with self.lock:
            jobs = []
            if self.current_job:
                jobs.append(self.current_job)
            jobs.extend(self.conversion_queue)
            jobs.extend(self.completed_jobs)

            # Clean up old completed jobs (older than 30 seconds)
            from datetime import datetime, timedelta

            cutoff = datetime.now() - timedelta(seconds=30)
            self.completed_jobs = [j for j in self.completed_jobs if j.completed_at and j.completed_at > cutoff]

            return jobs.copy()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job by ID."""
        with self.lock:
            # Check if it's the current job
            if self.current_job and self.current_job.id == job_id:
                self.current_job.status = ConversionStatus.CANCELLED
                logger.info(f"Marked current job {job_id} for cancellation")
                return True

            # Check queue
            for i, job in enumerate(self.conversion_queue):
                if job.id == job_id:
                    job.status = ConversionStatus.CANCELLED
                    self.conversion_queue.pop(i)
                    self._cleanup_temp_files(job)
                    self._notify_status_update(job)
                    logger.info(f"Cancelled queued job {job_id}")
                    return True

        return False

    def _worker_loop(self):
        """Main worker loop for processing conversions."""
        logger.info("Conversion worker loop started")

        while not self.shutdown_event.is_set():
            try:
                # Get next job from queue
                with self.lock:
                    if not self.conversion_queue:
                        job = None
                    else:
                        job = self.conversion_queue.pop(0)
                        self.current_job = job

                if job is None:
                    time.sleep(1)
                    continue

                # Process the job
                self._process_job(job)

                # Move to completed jobs if finished successfully or failed
                with self.lock:
                    if job.status in [ConversionStatus.COMPLETED, ConversionStatus.FAILED, ConversionStatus.CANCELLED]:
                        self.completed_jobs.append(job)
                        logger.info(f"Moved job {job.id} to completed jobs (status: {job.status})")
                    self.current_job = None

            except Exception as e:
                logger.error(f"Error in conversion worker loop: {e}")
                time.sleep(5)

        logger.info("Conversion worker loop stopped")

    def _process_job(self, job: ConversionJob):
        """Process a single conversion job."""
        logger.info(f"Processing conversion job: {job.id}")

        try:
            # Update status
            job.status = ConversionStatus.PROCESSING
            job.started_at = datetime.now()
            self._notify_status_update(job)

            # Check if cancelled
            if job.status == ConversionStatus.CANCELLED:
                self._cleanup_temp_files(job)
                return

            # Analyze input video
            input_info = self._analyze_video(job.input_path)
            if not input_info:
                job.status = ConversionStatus.FAILED
                job.error_message = "Failed to analyze input video"
                self._notify_status_update(job)
                self._cleanup_temp_files(job)
                return

            # Convert video
            if not self._convert_video(job, input_info):
                job.status = ConversionStatus.FAILED
                self._notify_status_update(job)
                self._cleanup_temp_files(job)
                return

            # Check if cancelled during conversion
            if job.status == ConversionStatus.CANCELLED:
                self._cleanup_temp_files(job)
                return

            # Validate output
            job.status = ConversionStatus.VALIDATING
            job.progress = 90.0
            self._notify_status_update(job)

            if not self._validate_output(job):
                job.status = ConversionStatus.FAILED
                job.error_message = "Output validation failed"
                self._notify_status_update(job)
                self._cleanup_temp_files(job)
                return

            # Move to final location and cleanup
            if not self._finalize_conversion(job):
                job.status = ConversionStatus.FAILED
                job.error_message = "Failed to finalize conversion"
                self._notify_status_update(job)
                self._cleanup_temp_files(job)
                return

            # Success
            job.status = ConversionStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = datetime.now()
            self._notify_status_update(job)

            logger.info(f"Conversion completed successfully: {job.id}")

        except Exception as e:
            logger.error(f"Error processing conversion job {job.id}: {e}")
            job.status = ConversionStatus.FAILED
            job.error_message = str(e)
            self._notify_status_update(job)
            self._cleanup_temp_files(job)

    def _analyze_video(self, video_path: Path) -> Optional[Dict]:
        """Analyze video properties using ffprobe."""
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"ffprobe failed: {result.stderr}")
                return None

            data = json.loads(result.stdout)

            # Find video stream
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if not video_stream:
                logger.error("No video stream found")
                return None

            info = {
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "duration": float(data.get("format", {}).get("duration", 0)),
                "fps": self._parse_fps(video_stream.get("r_frame_rate", "30/1")),
                "codec": video_stream.get("codec_name"),
                "pixel_format": video_stream.get("pix_fmt"),
            }

            logger.info(f"Video analysis: {info}")
            return info

        except Exception as e:
            logger.error(f"Failed to analyze video: {e}")
            return None

    def _parse_fps(self, fps_str: str) -> float:
        """Parse frame rate string like '30/1' to float."""
        try:
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                return num / den if den != 0 else 30.0
            return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return 30.0

    def _convert_video(self, job: ConversionJob, input_info: Dict) -> bool:
        """Convert video using FFmpeg."""
        try:
            config = self.config.get("output_format", {})

            # Calculate crop parameters
            crop_filter = self._calculate_crop_filter(
                input_info["width"], input_info["height"], config.get("width", 800), config.get("height", 480)
            )

            # Cap frame rate at 30fps - halve if higher
            input_fps = input_info["fps"]
            if input_fps > 30:
                output_fps = input_fps / 2
                logger.info(f"Input FPS {input_fps:.2f} > 30, converting to {output_fps:.2f} fps")
            else:
                output_fps = input_fps

            # Build FFmpeg command - input file is already in temp_conversions
            # Check that input file exists
            if not job.input_path.exists():
                logger.error(f"Input file does not exist: {job.input_path}")
                return False

            cmd = [
                "ffmpeg",
                "-y",
                "-stats",  # Enable stats output
                "-stats_period",
                "0.5",  # Update stats every 0.5 seconds
                "-i",
                str(job.input_path),
                "-vcodec",
                config.get("codec", "libx264"),
                "-profile:v",
                config.get("profile", "high"),
                "-level",
                config.get("level", "3.1"),
                "-pix_fmt",
                config.get("pixel_format", "yuv420p"),
                "-vf",
                f"{crop_filter},scale={config.get('width', 800)}:{config.get('height', 480)}",
                "-r",
                str(output_fps),
                "-preset",
                config.get("preset", "fast"),
                "-crf",
                str(config.get("crf", 23)),
                "-movflags",
                "+faststart",
            ]

            # Drop audio if configured
            if config.get("drop_audio", True):
                cmd.append("-an")

            cmd.append(str(job.temp_path))

            logger.info(f"FFmpeg command: {' '.join(cmd)}")

            # Start FFmpeg process with line-buffered I/O
            # Use stdbuf to disable buffering on FFmpeg's output
            buffered_cmd = ["stdbuf", "-oL", "-eL"] + cmd

            process = subprocess.Popen(
                buffered_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            # Monitor progress
            return self._monitor_ffmpeg_progress(job, process, input_info["duration"], input_info["fps"])

        except Exception as e:
            logger.error(f"Failed to convert video: {e}")
            return False

    def _calculate_crop_filter(self, input_width: int, input_height: int, target_width: int, target_height: int) -> str:
        """Calculate crop filter for center cropping to target aspect ratio."""
        target_ratio = target_width / target_height  # 800/480 = 5:3 ≈ 1.67
        input_ratio = input_width / input_height

        logger.info(
            f"Crop calculation: input {input_width}x{input_height} (ratio {input_ratio:.3f}) -> target {target_width}x{target_height} (ratio {target_ratio:.3f})"
        )

        if input_ratio > target_ratio:
            # Too wide, crop horizontally
            new_width = int(input_height * target_ratio)
            crop_x = (input_width - new_width) // 2
            return f"crop={new_width}:{input_height}:{crop_x}:0"
        else:
            # Too tall, crop vertically
            new_height = int(input_width / target_ratio)
            crop_y = (input_height - new_height) // 2
            return f"crop={input_width}:{new_height}:0:{crop_y}"

    def _monitor_ffmpeg_progress(
        self, job: ConversionJob, process: subprocess.Popen, total_duration: float, fps: float = 30.0
    ) -> bool:
        """Monitor FFmpeg progress and update job status."""
        # Calculate total expected frames
        total_frames = int(total_duration * fps)
        logger.info(
            f"Starting FFmpeg progress monitoring for job {job.id}, total duration: {total_duration:.2f}s, fps: {fps}, expected frames: {total_frames}"
        )
        try:
            stderr_output = []
            last_line = ""  # Track last line for carriage return handling

            # Read stderr line by line in real-time
            while True:
                # Check if cancelled
                if job.status == ConversionStatus.CANCELLED:
                    process.terminate()
                    process.wait(timeout=10)
                    return False

                # Read one line at a time (blocks until line available or EOF)
                line = process.stderr.readline()

                # If empty string, process has finished
                if not line:
                    break

                # Handle carriage return (\r) - FFmpeg uses this to update same line
                # Split on \r and take the last part (the most recent update)
                if "\r" in line:
                    parts = line.split("\r")
                    # Process all parts, last one is the current state
                    for part in parts:
                        if part.strip():
                            last_line = part

                    # Use the last (most recent) update
                    line = last_line
                else:
                    # Regular newline-terminated line
                    line = line.rstrip("\n")

                if line.strip():
                    stderr_output.append(line)

                    # Parse FFmpeg progress output - look for frame number
                    if "frame=" in line:
                        frame_match = re.search(r"frame=\s*(\d+)", line)
                        if frame_match:
                            current_frame = int(frame_match.group(1))
                            if total_frames > 0:
                                progress = min(85.0, (current_frame / total_frames) * 85.0)
                                # Only update if changed by at least 1%
                                if abs(progress - job.progress) >= 1.0:
                                    job.progress = progress
                                    logger.info(f"Progress: frame {current_frame}/{total_frames} = {progress:.1f}%")
                                    self._notify_status_update(job)

            # Wait for process to finish
            process.wait()

            # Check return code
            if process.returncode == 0:
                logger.info(f"FFmpeg completed successfully for job {job.id}")
                # Set to 85% and notify to ensure UI updates
                job.progress = 85.0
                self._notify_status_update(job)
                return True
            else:
                # Log the complete stderr output for debugging
                stderr_text = "\n".join(stderr_output)
                logger.error(f"FFmpeg failed with return code: {process.returncode}")
                logger.error(f"FFmpeg stderr output: {stderr_text[-1000:]}")  # Last 1000 chars
                job.error_message = f"FFmpeg failed: {stderr_text[-500:]}"  # Last 500 chars
                return False

        except Exception as e:
            logger.error(f"Error monitoring FFmpeg progress: {e}")
            return False

    def _validate_output(self, job: ConversionJob) -> bool:
        """Validate converted video meets requirements."""
        try:
            if not job.temp_path.exists():
                logger.error("Output file does not exist")
                return False

            # Analyze output
            output_info = self._analyze_video(job.temp_path)
            if not output_info:
                logger.error("Failed to analyze output video")
                return False

            config = self.config.get("output_format", {})
            validation = self.config.get("validation", {})

            # Check resolution
            if validation.get("check_resolution", True):
                expected_width = config.get("width", 800)
                expected_height = config.get("height", 480)
                if output_info["width"] != expected_width or output_info["height"] != expected_height:
                    logger.error(
                        f"Resolution mismatch: {output_info['width']}x{output_info['height']} != {expected_width}x{expected_height}"
                    )
                    return False

            # Check codec
            if validation.get("check_codec", True) and output_info["codec"] != "h264":
                logger.error(f"Codec mismatch: {output_info['codec']} != h264")
                return False

            # Check pixel format
            if validation.get("check_pixel_format", True):
                expected_pix_fmt = config.get("pixel_format", "yuv420p")
                if output_info["pixel_format"] != expected_pix_fmt:
                    logger.error(f"Pixel format mismatch: {output_info['pixel_format']} != {expected_pix_fmt}")
                    return False

            # Check no audio (if configured to drop audio)
            if validation.get("check_no_audio", True) and config.get("drop_audio", True):
                # Re-run ffprobe to check for audio streams
                cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(job.temp_path)]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
                    if audio_streams:
                        logger.error(f"Audio streams found when none expected: {len(audio_streams)}")
                        return False

            logger.info("Output validation passed")
            return True

        except Exception as e:
            logger.error(f"Failed to validate output: {e}")
            return False

    def _finalize_conversion(self, job: ConversionJob) -> bool:
        """Move converted file to final location and cleanup."""
        try:
            # Ensure uploads directory exists
            job.output_path.parent.mkdir(exist_ok=True)

            # Move converted file to uploads directory
            shutil.move(str(job.temp_path), str(job.output_path))
            logger.info(f"Moved converted file to: {job.output_path}")

            # Delete original uploaded file (which is in temp_conversions)
            if job.input_path.exists():
                job.input_path.unlink()
                logger.info(f"Deleted original temp file: {job.input_path}")

            # Cleanup any remaining temp files
            self._cleanup_temp_files(job)

            return True

        except Exception as e:
            logger.error(f"Failed to finalize conversion: {e}")
            return False

    def _cleanup_temp_files(self, job: ConversionJob):
        """Clean up temporary files for a job."""
        try:
            # Clean up input file (if it still exists - should already be deleted in finalize)
            if job.input_path.exists():
                job.input_path.unlink()
                logger.info(f"Cleaned up temp input: {job.input_path}")

            # Clean up temp output file (if it still exists)
            if job.temp_path.exists():
                job.temp_path.unlink()
                logger.info(f"Cleaned up temp output: {job.temp_path}")

        except Exception as e:
            logger.warning(f"Failed to cleanup temp files for job {job.id}: {e}")

    def _notify_status_update(self, job: ConversionJob):
        """Notify all status callbacks of job update."""
        logger.info(f"Notifying status update for job {job.id}: status={job.status}, progress={job.progress:.1f}%")
        for callback in self.status_callbacks:
            try:
                callback(job)
            except Exception as e:
                logger.warning(f"Status callback failed: {e}")


# Global conversion manager instance
conversion_manager: Optional[ConversionManager] = None


def get_conversion_manager() -> Optional[ConversionManager]:
    """Get the global conversion manager instance."""
    return conversion_manager


def initialize_conversion_manager(config: Dict):
    """Initialize the global conversion manager."""
    global conversion_manager

    if not config.get("video_conversion", {}).get("enabled", False):
        logger.info("Video conversion disabled in config")
        return

    conversion_config = config.get("video_conversion", {})
    conversion_manager = ConversionManager(conversion_config)
    conversion_manager.start()
    logger.info("Video conversion manager initialized and started")


def shutdown_conversion_manager():
    """Shutdown the global conversion manager."""
    global conversion_manager
    if conversion_manager:
        conversion_manager.stop()
        conversion_manager = None
        logger.info("Video conversion manager shutdown")
