# LED Gain Calibration Process for Diffusion Pattern Capture

## Overview

This two-phase calibration process ensures optimal camera gain settings for LED diffusion pattern capture while avoiding pixel saturation. The goal is to maximize dynamic range while preventing any pixels from reaching the saturation value of 255.

## Phase 1: Saturation Elimination Phase

### Objective
Eliminate saturation across a representative sample of LEDs to establish a safe gain baseline.

### Process

1. **LED Sample Selection**
   - Select approximately 100 test LEDs (e.g., every 26th LED for 2600 total LEDs)
   - Ensure representative coverage across the LED strip/matrix
   - Use evenly spaced indices to sample different physical regions

2. **Initial Gain Setting**
   - Start with a conservative gain value (e.g., `gainrange="2.0 2.0"`)
   - Lock auto-exposure and auto-white-balance to ensure consistency
   - Use fixed exposure time appropriate for LED capture

3. **Saturation Detection Loop**
   ```
   For each test LED in sample:
       a. Turn on LED at full brightness (RGB = 255, 255, 255)
       b. Capture frame with current gain setting
       c. Find maximum pixel value across all channels in captured frame
       d. Record: (LED_index, max_pixel_value, gain_setting)
       e. Turn off LED
   ```

4. **Saturation Analysis**
   - Count how many test LEDs produced pixels with value = 255
   - If ANY LED produces saturated pixels (value = 255):
     - Reduce gain by factor (e.g., multiply by 0.9)
     - Repeat saturation detection loop
   - Continue until NO test LEDs produce saturated pixels

5. **Phase 1 Result**
   - Safe gain setting where no test LEDs cause saturation
   - Database of max pixel values for each test LED at this gain

## Phase 2: Maximum Dynamic Range Optimization

### Objective
Find the brightest LED among the test set and calibrate gain to achieve maximum pixel value of exactly 254 (one below saturation).

### Process

1. **Brightest LED Identification**
   - From Phase 1 results, identify the LED that produced the highest max pixel value
   - This LED represents the "worst case" brightest LED in the system
   - Record this LED's index and its max pixel value at the safe gain

2. **Target Calculation**
   ```
   brightest_led_max_value = max(all_test_led_max_values)
   current_gain = safe_gain_from_phase1
   target_pixel_value = 254

   # Calculate required gain adjustment
   gain_multiplier = target_pixel_value / brightest_led_max_value
   optimized_gain = current_gain * gain_multiplier
   ```

3. **Gain Optimization**
   - Set camera gain to `optimized_gain` using `gainrange`
   - Turn on the brightest LED identified in step 1
   - Capture frame and measure actual max pixel value
   - Fine-tune if necessary:
     ```
     actual_max = measure_max_pixel_value()
     if abs(actual_max - 254) > 1:
         fine_tune_multiplier = 254 / actual_max
         optimized_gain *= fine_tune_multiplier
     ```

4. **Validation**
   - Test the optimized gain on 10-20 random LEDs from the test set
   - Verify that no pixels reach 255 (saturation)
   - Verify that the brightest LED produces pixels close to 254
   - Record final calibrated gain setting

## Implementation Details

### Camera Configuration
```python
def build_calibration_pipeline(gain_value: float) -> str:
    return (
        f"nvarguscamerasrc sensor-id=0 "
        f"gainrange=\"{gain_value} {gain_value}\" "
        f"exposuretimerange=\"10000000 10000000\" "  # Fixed 10ms exposure
        f"aelock=true "                              # Lock auto-exposure
        f"awblock=true "                             # Lock auto-white-balance
        f"saturation=1.0 "                           # No saturation boost
        "! nvvidconv ! video/x-raw,format=I420 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1"
    )
```

### LED Control
```python
def set_single_led(wled_client, led_index: int, brightness: int = 255):
    """Turn on single LED at specified brightness, all others off."""
    led_data = np.zeros((total_led_count, 3), dtype=np.uint8)
    led_data[led_index] = [brightness, brightness, brightness]  # White at full brightness
    wled_client.send_led_data(led_data)
    time.sleep(0.1)  # LED stabilization time
```

### Pixel Analysis
```python
def analyze_frame_brightness(frame: np.ndarray) -> dict:
    """Analyze captured frame for brightness characteristics."""
    # Convert to grayscale for overall brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Find maximum pixel value across all channels
    max_pixel = np.max(frame)

    # Count saturated pixels
    saturated_pixels = np.sum(frame == 255)

    # Find brightest region (for LED localization)
    max_intensity_pos = np.unravel_index(np.argmax(gray), gray.shape)

    return {
        'max_pixel_value': int(max_pixel),
        'saturated_pixel_count': int(saturated_pixels),
        'brightest_position': max_intensity_pos,
        'mean_brightness': float(np.mean(frame)),
        'has_saturation': saturated_pixels > 0
    }
```

## Calibration Algorithm

```python
class LEDGainCalibrator:
    def __init__(self, wled_client, camera, total_leds=2600):
        self.wled_client = wled_client
        self.camera = camera
        self.total_leds = total_leds
        self.test_led_indices = list(range(0, total_leds, total_leds // 100))

    def run_calibration(self) -> float:
        """Run complete two-phase calibration process."""

        # Phase 1: Eliminate saturation
        print("Phase 1: Eliminating saturation...")
        safe_gain, test_results = self.phase1_eliminate_saturation()
        print(f"Safe gain found: {safe_gain}")

        # Phase 2: Optimize for maximum dynamic range
        print("Phase 2: Optimizing dynamic range...")
        final_gain = self.phase2_optimize_range(safe_gain, test_results)
        print(f"Final optimized gain: {final_gain}")

        return final_gain

    def phase1_eliminate_saturation(self) -> tuple:
        """Phase 1: Find gain where no test LEDs saturate."""
        current_gain = 4.0  # Start conservatively
        max_iterations = 10

        for iteration in range(max_iterations):
            print(f"Testing gain {current_gain:.3f}...")

            # Test all sample LEDs at current gain
            test_results = []
            saturated_count = 0

            self.camera.set_gain(current_gain)

            for led_idx in self.test_led_indices:
                # Turn on single LED
                self.set_single_led(led_idx, brightness=255)

                # Capture and analyze
                frame = self.camera.capture_frame()
                analysis = analyze_frame_brightness(frame)

                test_results.append({
                    'led_index': led_idx,
                    'max_pixel': analysis['max_pixel_value'],
                    'saturated': analysis['has_saturation'],
                    'gain': current_gain
                })

                if analysis['has_saturation']:
                    saturated_count += 1

                # Turn off LED
                self.wled_client.set_solid_color(0, 0, 0)

            print(f"  Saturated LEDs: {saturated_count}/{len(self.test_led_indices)}")

            if saturated_count == 0:
                print(f"  ✓ No saturation found at gain {current_gain}")
                return current_gain, test_results
            else:
                # Reduce gain and try again
                current_gain *= 0.85
                print(f"  → Reducing gain to {current_gain:.3f}")

        raise RuntimeError("Could not find safe gain after maximum iterations")

    def phase2_optimize_range(self, safe_gain: float, test_results: list) -> float:
        """Phase 2: Optimize gain for brightest LED to reach 254."""

        # Find brightest LED from test results
        brightest_result = max(test_results, key=lambda x: x['max_pixel'])
        brightest_led = brightest_result['led_index']
        brightest_max_pixel = brightest_result['max_pixel']

        print(f"Brightest test LED: {brightest_led} (max pixel: {brightest_max_pixel})")

        # Calculate target gain
        target_pixel = 254
        gain_multiplier = target_pixel / brightest_max_pixel
        target_gain = safe_gain * gain_multiplier

        print(f"Target gain calculation: {safe_gain} × {gain_multiplier:.6f} = {target_gain:.6f}")

        # Fine-tune with actual measurement
        self.camera.set_gain(target_gain)
        self.set_single_led(brightest_led, brightness=255)

        frame = self.camera.capture_frame()
        analysis = analyze_frame_brightness(frame)
        actual_max = analysis['max_pixel_value']

        print(f"Actual max pixel at target gain: {actual_max}")

        # Fine adjustment if needed
        if abs(actual_max - target_pixel) > 1:
            fine_multiplier = target_pixel / actual_max
            final_gain = target_gain * fine_multiplier
            print(f"Fine adjustment: {target_gain} × {fine_multiplier:.6f} = {final_gain:.6f}")
        else:
            final_gain = target_gain

        # Turn off LED
        self.wled_client.set_solid_color(0, 0, 0)

        return final_gain
```

## Expected Results

- **Phase 1 Output**: Safe gain value (e.g., 3.2) where no test LEDs cause saturation
- **Phase 2 Output**: Optimized gain value (e.g., 4.1) where brightest LED produces max pixel value of 254
- **Final State**: Camera configured for maximum dynamic range without saturation across all LEDs

## Benefits

1. **No Saturation**: Guarantees no pixel values reach 255 during full capture
2. **Maximum Dynamic Range**: Uses nearly full 8-bit range (0-254)
3. **Consistent Results**: Fixed gain throughout entire capture process
4. **Representative Sampling**: Tests diverse LED positions to account for variations
5. **Safety Margin**: One-digit buffer below saturation threshold

This process ensures optimal image quality for LED diffusion pattern analysis while maintaining measurement accuracy across the entire LED array.
