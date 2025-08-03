# Makefile for ahead-of-time compilation of CUDA MMA kernels
#
# This compiles the batch MMA kernels to a shared library that can be
# loaded by Python at runtime, avoiding runtime compilation issues.

# CUDA Toolkit Configuration
NVCC = nvcc

# Compiler flags for CUDA module (.ptx for CuPy compatibility)
NVCC_FLAGS = -ptx
NVCC_FLAGS += --use_fast_math -O3
NVCC_FLAGS += -arch=sm_87  # Target exact GPU architecture (8.7)
NVCC_FLAGS += -DMMA_ENABLED

# Directories
SRC_DIR = .
BUILD_DIR = build
TARGET_DIR = .

# Source and target files
KERNEL_SRC = $(SRC_DIR)/batch_mma_kernel.cu
KERNEL_MODULE = $(TARGET_DIR)/batch_mma_kernel.ptx

# Python library information (for include paths)
PYTHON_INCLUDE = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
NUMPY_INCLUDE = $(shell python3 -c "import numpy; print(numpy.get_include())")

# Default target
all: $(KERNEL_MODULE)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile CUDA kernel to CUBIN module
$(KERNEL_MODULE): $(KERNEL_SRC) | $(BUILD_DIR)
	@echo "Compiling CUDA MMA kernels to PTX module..."
	@echo "CUDA Architecture: sm_87"
	@echo "NVCC Flags: $(NVCC_FLAGS)"
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "Successfully compiled: $@"

# Check GPU compute capability
check-gpu:
	@echo "Checking GPU compute capability..."
	@python3 -c "import cupy; device = cupy.cuda.Device(); cap = device.compute_capability; print(f'GPU Compute Capability: {cap[0]}.{cap[1]}'); print('GPU supports MMA operations' if int(cap[0]) >= 8 else 'WARNING: GPU compute capability < 8.0')"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(KERNEL_MODULE)

# Install dependencies (if needed)
install-deps:
	@echo "Checking CUDA toolkit installation..."
	@which nvcc || (echo "ERROR: nvcc not found. Please install CUDA toolkit."; exit 1)
	@echo "CUDA toolkit found: $$(nvcc --version | grep release)"

# Test compilation
test: $(KERNEL_MODULE)
	@echo "Testing kernel module..."
	@python3 -c "import os; print('✓ Kernel module exists') if os.path.exists('$(KERNEL_MODULE)') else print('✗ Kernel module missing')"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Compile CUDA kernels (default)"
	@echo "  check-gpu    - Check GPU compute capability"
	@echo "  clean        - Remove build artifacts"
	@echo "  install-deps - Check dependencies"
	@echo "  test         - Test compiled library"
	@echo "  help         - Show this help"

.PHONY: all clean check-gpu install-deps test help
