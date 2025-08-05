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
KERNEL8_CORRECTED_SRC = $(SRC_DIR)/batch8_corrected_kernel.cu
KERNEL8_EXPERIMENTAL_SRC = $(SRC_DIR)/batch8_experimental_kernel.cu
KERNEL_MODULE = $(TARGET_DIR)/batch_mma_kernel.ptx
KERNEL8_CORRECTED_MODULE = $(TARGET_DIR)/batch8_corrected_kernel.ptx
KERNEL8_EXPERIMENTAL_MODULE = $(TARGET_DIR)/batch8_experimental_kernel.ptx

# Python library information (for include paths)
PYTHON_INCLUDE = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
NUMPY_INCLUDE = $(shell python3 -c "import numpy; print(numpy.get_include())")

# Default target
all: $(KERNEL_MODULE) $(KERNEL8_CORRECTED_MODULE) $(KERNEL8_EXPERIMENTAL_MODULE)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile CUDA kernel to PTX module
$(KERNEL_MODULE): $(KERNEL_SRC) | $(BUILD_DIR)
	@echo "Compiling CUDA MMA kernels to PTX module..."
	@echo "CUDA Architecture: sm_87"
	@echo "NVCC Flags: $(NVCC_FLAGS)"
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "Successfully compiled: $@"

# Compile 8-frame corrected CUDA kernel to PTX module
$(KERNEL8_CORRECTED_MODULE): $(KERNEL8_CORRECTED_SRC) | $(BUILD_DIR)
	@echo "Compiling 8-frame corrected CUDA MMA kernels to PTX module..."
	@echo "CUDA Architecture: sm_87"
	@echo "NVCC Flags: $(NVCC_FLAGS)"
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "Successfully compiled: $@"

# Compile 8-frame experimental CUDA kernel to PTX module
$(KERNEL8_EXPERIMENTAL_MODULE): $(KERNEL8_EXPERIMENTAL_SRC) | $(BUILD_DIR)
	@echo "Compiling 8-frame experimental CUDA MMA kernels to PTX module..."
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
	rm -f $(KERNEL_MODULE) $(KERNEL8_CORRECTED_MODULE) $(KERNEL8_EXPERIMENTAL_MODULE)

# Install dependencies (if needed)
install-deps:
	@echo "Checking CUDA toolkit installation..."
	@which nvcc || (echo "ERROR: nvcc not found. Please install CUDA toolkit."; exit 1)
	@echo "CUDA toolkit found: $$(nvcc --version | grep release)"

# Test compilation
test: $(KERNEL_MODULE) $(KERNEL8_CORRECTED_MODULE) $(KERNEL8_EXPERIMENTAL_MODULE)
	@echo "Testing kernel modules..."
	@python3 -c "import os; print('✓ 16-frame kernel module exists') if os.path.exists('$(KERNEL_MODULE)') else print('✗ 16-frame kernel module missing')"
	@python3 -c "import os; print('✓ 8-frame corrected kernel module exists') if os.path.exists('$(KERNEL8_CORRECTED_MODULE)') else print('✗ 8-frame corrected kernel module missing')"
	@python3 -c "import os; print('✓ 8-frame experimental kernel module exists') if os.path.exists('$(KERNEL8_EXPERIMENTAL_MODULE)') else print('✗ 8-frame experimental kernel module missing')"

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
