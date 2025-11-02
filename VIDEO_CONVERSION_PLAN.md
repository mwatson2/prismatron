# Video Conversion System Implementation Plan

## Overview
Implement an automatic video conversion system to convert uploaded videos to H.264/800x480/8-bit format optimized for hardware decoding on the Jetson platform.

## Current Upload Flow Analysis
- Files uploaded via `/api/upload` endpoint
- Saved to `uploads/` directory with UUID filename
- Immediately added to playlist and managed playlist
- No processing or validation of video properties

## Requirements
1. **Target Format**: H.264, 800x480, 8-bit color depth, original frame rate
2. **Aspect Ratio Handling**: Crop (center crop) rather than scale/letterbox
3. **Hardware Decoder Compatibility**: Ensure output works with `h264_nvv4l2dec`
4. **Queue Management**: One conversion at a time
5. **Progress Tracking**: Real-time progress updates via WebSocket
6. **UI Integration**: Progress list on Upload page

## System Architecture

### 1. Conversion Queue System
```
ConversionManager
├── conversion_queue: List[ConversionJob]
├── current_job: Optional[ConversionJob]
├── worker_thread: threading.Thread
└── status_callbacks: List[Callable]
```

### 2. ConversionJob Model
```python
@dataclass
class ConversionJob:
    id: str                    # UUID for tracking
    input_path: Path          # Original file path in uploads/
    temp_path: Path           # Temporary file path during conversion
    output_path: Path         # Final converted file path in uploads/
    original_name: str        # Original filename
    status: ConversionStatus  # QUEUED, PROCESSING, VALIDATING, COMPLETED, FAILED
    progress: float           # 0.0 to 100.0
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_duration: Optional[float]
```

### 3. ConversionStatus Enum
```python
class ConversionStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

## Implementation Components

### A. Backend Components

#### 1. Video Conversion Module (`src/web/video_converter.py`)
- `ConversionManager` class
- `ConversionJob` dataclass  
- `ConversionStatus` enum
- FFmpeg conversion logic with progress monitoring
- Thread-safe queue management

#### 2. FFmpeg Conversion Parameters
```bash
ffmpeg -i input.mp4 \
  -vcodec libx264 \
  -profile:v high \
  -level 3.1 \
  -pix_fmt yuv420p \
  -vf "scale=800:480:force_original_aspect_ratio=increase,crop=800:480" \
  -r {original_fps} \
  -preset fast \
  -crf 23 \
  -an \
  -movflags +faststart \
  temp_conversions/{job_id}_output.mp4
```
**Note**: Audio is completely dropped using `-an` flag as requested.

#### 3. API Endpoints
- `GET /api/conversions` - List active/recent conversions
- `POST /api/conversions/{job_id}/cancel` - Cancel conversion
- `DELETE /api/conversions/{job_id}` - Remove completed job from list
- WebSocket updates for real-time progress

#### 4. Modified Upload Flow
```python
# In upload_file() endpoint:
if content_type == "video":
    # Don't add to playlist immediately
    # Queue for conversion instead
    conversion_job = await conversion_manager.queue_conversion(file_path, original_name)
    return {"status": "queued_for_conversion", "job_id": conversion_job.id}
else:
    # Images processed immediately as before
    ...
```

#### 5. Conversion Process Flow
1. **Upload**: Video saved to `uploads/` directory
2. **Queue**: Job added to conversion queue
3. **Process**: FFmpeg converts to `temp_conversions/` directory
4. **Validate**: Check output file properties with ffprobe
5. **Finalize**: Move validated file to `uploads/`, delete original
6. **Playlist**: Add converted file to playlist automatically

### B. Frontend Components

#### 1. ConversionProgress Component (`src/web/frontend/src/components/ConversionProgress.jsx`)
- Real-time progress display
- Cancel/retry functionality
- Status indicators
- ETA display

#### 2. Modified UploadPage.jsx
- Integration with ConversionProgress component
- Upload status differentiation (immediate vs queued)
- Conversion queue display

#### 3. WebSocket Integration
- Listen for conversion progress updates
- Update UI in real-time
- Handle conversion completion/failure

### C. Conversion Logic Details

#### 1. Video Analysis (Input)
- Use `ffprobe` to detect:
  - Current resolution
  - Aspect ratio
  - Frame rate
  - Color depth
  - Codec information

#### 2. Output Validation
- Use `ffprobe` to verify converted file:
  - Resolution exactly 800x480
  - Codec is H.264
  - Pixel format is yuv420p (8-bit)
  - No audio streams present
  - File is not corrupted
- If validation fails, mark job as FAILED and retry

#### 3. Crop Calculation
```python
def calculate_crop_params(width: int, height: int) -> tuple:
    """Calculate crop parameters for 800x480 center crop."""
    target_ratio = 800 / 480  # 5:3
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Too wide, crop horizontally
        new_width = int(height * target_ratio)
        crop_x = (width - new_width) // 2
        return f"crop={new_width}:{height}:{crop_x}:0"
    else:
        # Too tall, crop vertically  
        new_height = int(width / target_ratio)
        crop_y = (height - new_height) // 2
        return f"crop={width}:{new_height}:0:{crop_y}"
```

#### 4. Progress Monitoring
- Parse FFmpeg stderr output for progress
- Extract time/duration information
- Calculate percentage completion
- Broadcast via WebSocket

#### 5. Error Handling
- FFmpeg process failures
- Disk space issues
- Invalid input formats
- Network interruptions

## File Organization

```
src/web/
├── video_converter.py          # Core conversion logic
├── api_server.py              # Modified upload endpoints
└── frontend/src/
    ├── components/
    │   └── ConversionProgress.jsx
    ├── pages/
    │   └── UploadPage.jsx      # Modified
    └── hooks/
        └── useConversions.js   # WebSocket hook

# New directory structure
temp_conversions/               # Temporary conversion workspace
├── {job_id}_input.ext         # Copy of original during processing
└── {job_id}_output.mp4        # Converted output before validation
```

## State Storage
- **In-memory only**: Conversion queue stored in memory (resets on restart)
- No database or persistence needed as requested
- WebSocket broadcasts for real-time updates

## Configuration (config.json)
Add new video conversion section to existing config.json:
```json
{
  "video_conversion": {
    "enabled": true,
    "max_concurrent_jobs": 1,
    "temp_directory": "temp_conversions",
    "output_format": {
      "codec": "libx264",
      "profile": "high",
      "level": "3.1",
      "pixel_format": "yuv420p",
      "width": 800,
      "height": 480,
      "preset": "fast",
      "crf": 23,
      "drop_audio": true
    },
    "validation": {
      "check_resolution": true,
      "check_codec": true,
      "check_pixel_format": true,
      "check_no_audio": true
    }
  }
}
```

## Integration Points

### 1. Upload Flow Modification
- Video uploads → conversion queue
- Image uploads → immediate processing (unchanged)
- Return different response for video vs image uploads

### 2. Playlist Integration  
- Add converted video to playlist only after successful conversion
- Update playlist item with converted file path
- Notify via WebSocket when video is ready

### 3. File Management
- **Delete original after successful conversion**: Original uploaded file removed after validation passes
- **Cleanup converted files**: 24h cleanup applies to final converted files in uploads/
- **Temp directory cleanup**: Failed/cancelled jobs cleaned from temp_conversions/
- **Handle orphaned jobs**: Clean up temp files for jobs that fail to complete

## Testing Strategy
1. **Unit Tests**: Conversion logic, crop calculations
2. **Integration Tests**: Full upload→conversion→playlist flow  
3. **Performance Tests**: Multiple file handling, large files
4. **Error Tests**: Invalid files, process failures

## Phase 1 Implementation Order
1. ✅ Add video conversion config to config.json - **COMPLETED**
2. ✅ Create `temp_conversions/` directory and ensure permissions - **COMPLETED**
3. ✅ Create `ConversionManager` and related classes with validation - **COMPLETED**
4. ✅ Implement FFmpeg conversion with progress monitoring - **COMPLETED**
5. ✅ Add validation step using ffprobe - **COMPLETED**
6. ✅ Add API endpoints for conversion management - **COMPLETED**
7. ✅ Modify upload endpoint to queue video conversions - **COMPLETED**
8. ✅ Create frontend ConversionProgress component - **COMPLETED**
9. ✅ Integrate WebSocket updates - **COMPLETED**
10. ✅ Update UploadPage to show conversion status - **COMPLETED**

## Implementation Progress
**Current Status**: ✅ **PHASE 1 COMPLETE** - All core functionality implemented!

### Completed Components:

#### Backend Implementation:
- ✅ **Configuration**: Complete video conversion config in config.json
- ✅ **Directory Structure**: `temp_conversions/` directory with proper permissions
- ✅ **ConversionManager**: Complete class with:
  - Thread-safe job queue management
  - Single-threaded conversion processing
  - FFmpeg integration with real-time progress monitoring
  - Comprehensive output validation using ffprobe
  - Automatic file management (copy→convert→validate→move→cleanup→delete original)
- ✅ **ConversionJob**: Complete dataclass with all required tracking fields
- ✅ **ConversionStatus**: Enum including VALIDATING state
- ✅ **FFmpeg Integration**:
  - Audio completely dropped (-an flag)
  - Center crop calculation for 5:3 aspect ratio conversion
  - H.264/800x480/8-bit output format
  - Progress monitoring via stderr parsing
- ✅ **API Endpoints**: Full REST API for conversion management
- ✅ **Upload Integration**: Videos queued for conversion instead of immediate playlist addition
- ✅ **WebSocket Broadcasting**: Real-time conversion status updates
- ✅ **Error Handling**: Comprehensive logging and error recovery

#### Frontend Implementation:
- ✅ **ConversionProgress Component**: Real-time progress display with:
  - Status indicators and progress bars
  - Cancel/remove functionality  
  - Error message display
  - Conversion queue summary
- ✅ **useConversions Hook**: Custom React hook for:
  - API integration
  - Real-time polling for active conversions
  - WebSocket update handling
  - Conversion management actions
- ✅ **UploadPage Integration**:
  - Separate handling for images vs videos
  - Conversion progress display
  - Updated upload guidelines
  - Enhanced status messaging

## Future Enhancements
- Batch conversion support
- Quality presets (fast/balanced/high quality)
- Custom resolution targets
- Video preview generation
- Conversion history/logs
- Resume interrupted conversions

## Success Criteria
- Videos automatically converted to hardware-compatible format (H.264/800x480/8-bit)
- Audio completely removed from output files
- Real-time progress tracking in UI with validation step
- Single-threaded conversion queue working from temp directory
- **Original uploaded files deleted after successful conversion**
- **Output validation ensures correct format before playlist addition**
- Converted videos play correctly with `h264_nvv4l2dec`
- Temp directory properly managed (creation/cleanup)
- Configuration stored in global config.json
- No manual intervention required for typical uploads
