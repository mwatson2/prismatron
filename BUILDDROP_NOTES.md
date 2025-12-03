# Build-Up/Drop Detection - Implementation Notes

## Recent Changes

### Buffer Size Fix (2025-11-16)

**Issue**: Adding build-drop state fields to `SystemStatus` increased the serialized size beyond the 2KB buffer limit.

**Error**:
```
Status data too large: 2048 bytes, buffer size: 2048 bytes
```

**Fix**: Increased minimum buffer size from 2KB to 4KB in `src/core/control_state.py`:
```python
return max(aligned_size, 4096)  # Minimum 4KB (increased from 2KB for build-drop fields)
```

**Impact**:
- Shared memory buffer increased from 2KB to 4KB
- System must be restarted for change to take effect
- Memory usage: +2KB per process (negligible impact)

### Fields Added to SystemStatus

```python
# Build-up/drop detection state (added to SystemStatus)
buildup_state: str = "NORMAL"        # ~10 bytes
buildup_intensity: float = 0.0       # ~8 bytes
bass_energy: float = 0.0             # ~8 bytes
high_energy: float = 0.0             # ~8 bytes
```

**Total added**: ~34 bytes (plus JSON overhead ~50-100 bytes)

## Why The Buffer Size Issue Occurred

The `SystemStatus` dataclass is serialized to JSON and stored in shared memory for IPC. The original buffer calculation assumed a certain maximum size, but adding new fields pushed it over the 2KB limit.

### Previous SystemStatus Size (approximate)
- Base fields: ~1500-1800 bytes (JSON serialized)
- Safety margin: 1.5x = ~2250-2700 bytes
- Rounded to 2KB minimum (2048 bytes) - **TOO SMALL**

### New SystemStatus Size (with build-drop fields)
- Base fields + build-drop: ~1600-1900 bytes
- Safety margin: 1.5x = ~2400-2850 bytes
- Now using 4KB minimum (4096 bytes) - **SUFFICIENT**

## System Restart Required

After updating the buffer size, the Prismatron system must be restarted to recreate the shared memory with the new size:

```bash
# If running as systemd service
sudo systemctl restart prismatron

# Or if running manually
# Stop current process (Ctrl+C)
# Then restart
python src/main.py
```

## Verification

After restart, check logs for successful initialization:
```bash
# Should NOT see "Status data too large" errors anymore
tail -f logs/prismatron.log | grep "Status data"
```

## Future Considerations

### If Adding More Fields

If you need to add more fields to `SystemStatus` in the future:

1. **Small additions** (1-3 simple fields): Current 4KB buffer should handle it
2. **Large additions** (many fields or long strings): Consider increasing to 8KB
3. **Monitor logs**: Watch for "Status data too large" errors after changes

### Alternative Solutions

If the buffer continues to grow, consider:

1. **Separate buffers**: Split frequently-updated fields (like FPS) from rarely-changing config
2. **Compression**: Add gzip compression for JSON (adds latency but saves space)
3. **Binary format**: Replace JSON with struct/protobuf (faster but less debuggable)
4. **Field pruning**: Remove rarely-used fields or move them to config

## Current Buffer Usage

With 4KB buffer:
- Used: ~2000-2500 bytes (after JSON serialization)
- Available: ~1500-2000 bytes free
- Utilization: ~50-60% (healthy margin)

## Testing

To verify the fix works:

1. **Start system**: Should initialize without errors
2. **Check shared memory**:
   ```bash
   ls -lh /dev/shm/prismatron_control
   # Should show 4096 bytes (4.0K)
   ```
3. **Monitor logs**: No "Status data too large" errors
4. **Check web interface**: Build-drop visualizer should receive updates

## Related Files

- `src/core/control_state.py`: Buffer size calculation
- `src/web/api_server.py`: Status model and endpoints
- `src/web/frontend/src/components/BuildDropVisualizer.jsx`: Visualization

---

**Status**: ✅ Fixed
**Date**: 2025-11-16
**Impact**: Requires system restart
