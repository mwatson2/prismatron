# Debug Tools

This directory contains debug tools and test utilities created during development of the Prismatron LED Display System. These tools are preserved for future reference and debugging purposes.

## Files

### Debug Tools

- **`debug_web_preview.py`** - Web interface preview debugging tool
  - Retrieves LED positions from web API and compares with diffusion patterns
  - Connects to WebSocket API to receive LED values
  - Renders LED values using preview-style techniques for debugging
  - Usage: `python debug_web_preview.py --patterns patterns.npz --api-url http://localhost:8000`

- **`test_frontend_rendering.py`** - Frontend rendering test tool
  - Replicates exact rendering approach used by HomePage.jsx frontend component
  - Tests LED visualization with additive blending, glow effects, and coordinate scaling
  - Helps diagnose rendering differences between Python and JavaScript
  - Usage: `python test_frontend_rendering.py --patterns patterns.npz --api-url http://localhost:8000`

### Test Scripts and Data

- **`test_checkerboard.py`** - Test pattern generator
  - Creates 800x480 checkerboard test image with 5x3 grid pattern
  - Black/white squares with center RGB color squares for testing
  - Generates test pattern for LED optimization validation

- **`test_checkerboard_reconstructed.png`** - Test output image
  - Generated output from test_checkerboard.py
  - Reference image for visual testing and validation

- **`debug_mixed_led_values.npz`** - Debug data file
  - Contains LED value data captured during debugging sessions
  - Preserved for reproducing specific debugging scenarios

## Purpose

These tools were created to:

1. **Debug web interface issues** - Identify rendering discrepancies between frontend and backend
2. **Test LED visualization** - Validate LED position mapping and color rendering
3. **Generate test patterns** - Create known test images for optimization validation
4. **Preserve debugging context** - Keep debugging data and tools for future reference

## Usage Notes

- Most tools require the system to be running with web server and WebSocket API available
- Pattern files (`.npz`) from `diffusion_patterns/` directory are typically required
- Tools use PIL/Pillow for image generation and manipulation
- Some tools connect to WebSocket API at `ws://localhost:8000/ws` by default

## Historical Context

These tools were created during the development of:
- Web interface preview rendering with brightness factors
- LED position coordinate mapping
- Frontend/backend rendering consistency
- WebSocket real-time preview functionality

They serve as references for understanding the debugging process and can be adapted for future debugging needs.
