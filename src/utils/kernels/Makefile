# Makefile for ahead-of-time compilation of CUDA MMA kernels
#
# This compiles the batch MMA kernels to PTX files for multiple GPU architectures
# that can be loaded by Python at runtime via CuPy.

# CUDA Toolkit Configuration
NVCC = nvcc

# Base compiler flags
NVCC_BASE_FLAGS = -ptx --use_fast_math -O3 -DMMA_ENABLED

# Target GPU architectures (compile for all common datacenter/consumer GPUs)
# sm_80: Ampere (A100)
# sm_86: Ampere (A10G, A40, RTX 3000 series) - common in AWS
# sm_87: Jetson Orin
# sm_89: Ada Lovelace (L4, L40, RTX 4000 series)
GPU_ARCHS = sm_80 sm_86 sm_87 sm_89

# Directories
SRC_DIR = .
BUILD_DIR = build
TARGET_DIR = .

# Source files
KERNEL_SRC = $(SRC_DIR)/batch_mma_kernel.cu
KERNEL8_CORRECTED_SRC = $(SRC_DIR)/batch8_corrected_kernel.cu
KERNEL8_EXPERIMENTAL_SRC = $(SRC_DIR)/batch8_experimental_kernel.cu

# Generate PTX filenames for each architecture
KERNEL_MODULES = $(foreach arch,$(GPU_ARCHS),$(TARGET_DIR)/batch_mma_kernel_$(arch).ptx)
KERNEL8_CORRECTED_MODULES = $(foreach arch,$(GPU_ARCHS),$(TARGET_DIR)/batch8_corrected_kernel_$(arch).ptx)
KERNEL8_EXPERIMENTAL_MODULES = $(foreach arch,$(GPU_ARCHS),$(TARGET_DIR)/batch8_experimental_kernel_$(arch).ptx)

# Default target - compile for all architectures
all: $(KERNEL_MODULES) $(KERNEL8_CORRECTED_MODULES) $(KERNEL8_EXPERIMENTAL_MODULES)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Pattern rule for batch_mma_kernel
$(TARGET_DIR)/batch_mma_kernel_sm_%.ptx: $(KERNEL_SRC) | $(BUILD_DIR)
	@echo "Compiling batch_mma_kernel for sm_$*..."
	$(NVCC) $(NVCC_BASE_FLAGS) -arch=sm_$* -o $@ $<
	@echo "Successfully compiled: $@"

# Pattern rule for batch8_corrected_kernel
$(TARGET_DIR)/batch8_corrected_kernel_sm_%.ptx: $(KERNEL8_CORRECTED_SRC) | $(BUILD_DIR)
	@echo "Compiling batch8_corrected_kernel for sm_$*..."
	$(NVCC) $(NVCC_BASE_FLAGS) -arch=sm_$* -o $@ $<
	@echo "Successfully compiled: $@"

# Pattern rule for batch8_experimental_kernel
$(TARGET_DIR)/batch8_experimental_kernel_sm_%.ptx: $(KERNEL8_EXPERIMENTAL_SRC) | $(BUILD_DIR)
	@echo "Compiling batch8_experimental_kernel for sm_$*..."
	$(NVCC) $(NVCC_BASE_FLAGS) -arch=sm_$* -o $@ $<
	@echo "Successfully compiled: $@"

# Check GPU compute capability
check-gpu:
	@echo "Checking GPU compute capability..."
	@python3 -c "import cupy; device = cupy.cuda.Device(); cap = device.compute_capability; print(f'GPU Compute Capability: {cap[0]}.{cap[1]}'); print('GPU supports MMA operations' if int(cap[0]) >= 8 else 'WARNING: GPU compute capability < 8.0')"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET_DIR)/batch_mma_kernel_sm_*.ptx
	rm -f $(TARGET_DIR)/batch8_corrected_kernel_sm_*.ptx
	rm -f $(TARGET_DIR)/batch8_experimental_kernel_sm_*.ptx
	# Also clean old single-arch files if they exist
	rm -f $(TARGET_DIR)/batch_mma_kernel.ptx
	rm -f $(TARGET_DIR)/batch8_corrected_kernel.ptx
	rm -f $(TARGET_DIR)/batch8_experimental_kernel.ptx

# Install dependencies (if needed)
install-deps:
	@echo "Checking CUDA toolkit installation..."
	@which nvcc || (echo "ERROR: nvcc not found. Please install CUDA toolkit."; exit 1)
	@echo "CUDA toolkit found: $$(nvcc --version | grep release)"

# Test compilation - list all compiled PTX files
test: all
	@echo "Testing kernel modules..."
	@echo "Compiled PTX files:"
	@ls -la $(TARGET_DIR)/*.ptx 2>/dev/null || echo "No PTX files found"
	@echo ""
	@echo "Architecture coverage:"
	@for arch in $(GPU_ARCHS); do \
		if [ -f "$(TARGET_DIR)/batch_mma_kernel_$$arch.ptx" ]; then \
			echo "✓ $$arch: batch_mma_kernel"; \
		else \
			echo "✗ $$arch: batch_mma_kernel MISSING"; \
		fi; \
		if [ -f "$(TARGET_DIR)/batch8_corrected_kernel_$$arch.ptx" ]; then \
			echo "✓ $$arch: batch8_corrected_kernel"; \
		else \
			echo "✗ $$arch: batch8_corrected_kernel MISSING"; \
		fi; \
		if [ -f "$(TARGET_DIR)/batch8_experimental_kernel_$$arch.ptx" ]; then \
			echo "✓ $$arch: batch8_experimental_kernel"; \
		else \
			echo "✗ $$arch: batch8_experimental_kernel MISSING"; \
		fi; \
	done

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Compile CUDA kernels for all architectures: $(GPU_ARCHS)"
	@echo "  check-gpu    - Check GPU compute capability"
	@echo "  clean        - Remove build artifacts"
	@echo "  install-deps - Check dependencies"
	@echo "  test         - Test compiled PTX files"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Supported GPU architectures:"
	@echo "  sm_80 - Ampere (A100)"
	@echo "  sm_86 - Ampere (A10G, A40, RTX 3000)"
	@echo "  sm_87 - Jetson Orin"
	@echo "  sm_89 - Ada Lovelace (L4, L40, RTX 4000)"

.PHONY: all clean check-gpu install-deps test help
