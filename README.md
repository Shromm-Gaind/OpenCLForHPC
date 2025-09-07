# OpenCL for HPC

This code was prepared for the OpenCL for HPC presentation to showcase various optimization techniques in high-performance computing.

Written by Shromm Gaind contact me at sgaind88@gmail.com or darth_creamtor on Discord

The presentation slides are provided on this github repo as a pdf

Check out the stream on https://www.youtube.com/watch?v=OJX8baW2s3U

This code is under the MIT licence 

## The Optimization Stages

The project is organized into folders, each representing a distinct stage in the optimization journey.

*   **`01-sequential-cpu`**: The baseline implementation. A single-threaded, pure C program that runs entirely on the CPU. This serves as our performance reference point.

*   **`02-naive-opencl`**: The first step to GPU acceleration. The algorithm is ported to OpenCL using two simple kernels, one for the row updates and one for the column updates. The main loop still synchronizes at every step (`k`), limiting performance.

*   **`03-block-hybrid`**: A significant algorithmic change. Instead of operating on single rows/columns, we use a block decomposition (or "panel factorization") method. A small diagonal panel is processed on the CPU, while the much larger trailing matrix updates (TRSM and GEMM operations) are performed by more efficient kernels on the GPU.

*   **`04-async-vectorized`**: incorporates several key optimizations:
    *   **Asynchronous Execution**: Uses two command queues to overlap data transfers (CPU<->GPU) with GPU computation, hiding memory latency.
    *   **Pinned Memory**: Improves data transfer speeds between the host and device.
    *   **Optimized Kernels**: The GEMM kernel is rewritten using register blocking and vectorization (`float4`) to maximize arithmetic intensity and memory bandwidth, achieving  higher GFLOPS.

## Setting up the Prerequisites

Before building, you need to ensure you have a C++ compiler, CMake, and the correct OpenCL SDK for your hardware.

### 1. C++ Compiler and CMake

*   **A C/C++ Compiler**:
    *   **Linux**: `sudo apt install build-essential` or equivalent for your distribution.
    *   **Windows**: Install the "Desktop development with C++" workload from the Visual Studio Installer.
    *   **macOS**: Install the Xcode Command Line Tools: `xcode-select --install`.
*   **CMake**: (Version 3.15 or newer). You can download it from the [official CMake website](https://cmake.org/download/) or install it via a package manager.

### 2. OpenCL SDK Installation

Installing the OpenCL SDK is hardware-specific. You need to get the correct drivers and development files from your GPU vendor.

#### For NVIDIA GPUs:
1.  **Install the CUDA Toolkit**: The full CUDA Toolkit includes the NVIDIA OpenCL driver, compiler (for PTX), and the necessary headers and libraries.
2.  **Download Page**: [NVIDIA CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
3.  **Linux**: After installation, you may also need to install the OpenCL ICD loader:
    ```bash
    # For Debian/Ubuntu
    sudo apt install ocl-icd-opencl-dev

    # For Arch Linux
    sudo pacman -S ocl-icd
    ```

#### For AMD GPUs:
1.  **Install AMD ROCm or the AMD Adrenalin Software**: For modern GPUs, the ROCm platform is recommended. For older consumer cards, the standard driver package may suffice.
2.  **Download Page**: [AMD ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) or [AMD Drivers and Support](https://www.amd.com/en/support).
3.  **Linux**: The installation scripts provided by AMD usually handle all dependencies.

#### For Intel GPUs / CPUs:
1.  **Install the Intel oneAPI Base Toolkit**: This toolkit contains all the necessary components for OpenCL development on Intel hardware (integrated GPUs, discrete GPUs, and CPUs).
2.  **Download Page**: [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)

#### Verifying the Installation
After installing the SDK, you can verify that your system recognizes the OpenCL devices by installing and running `clinfo`.

```bash
# For Debian/Ubuntu
sudo apt install clinfo
clinfo

# For Arch Linux
sudo pacman -S clinfo
clinfo
````

If the installation was successful, clinfo will output a list of available platforms (e.g., NVIDIA CUDA, AMD, Intel) and the devices within them. If you see your GPU listed, you are ready to build the project.
## How to Build

This project uses a top-level CMake file to build all stages at once.

```bash
# 1. Clone the repository
git clone https://github.com/Shromm-Gaind/OpenCLForHPC.git
cd lu-decomposition-showcase

# 2. Create a build directory
mkdir build
cd build

# 3. Configure the project
cmake ..

# 4. Build all executables
cmake --build .
```

## How to Run

After building, the executables for each stage will be located in subfolders within the `build` directory.

```bash
# From the 'build' directory:

# Run the CPU baseline
./01-sequential-cpu/sequential_cpu

# Run the naive GPU version
./02-naive-opencl/naive_opencl

# Run the block-hybrid version
./03-block-hybrid/block_hybrid

# Run the final asynchronous/vectorized version
./04-async-vectorized/async_vectorized
```

---

### Extras: GEMM Benchmark

The `extras/gemm-benchmark` directory contains a standalone harness for testing and comparing different GEMM (General Matrix Multiplication) OpenCL kernels. This was used to develop the high-performance kernel used in the final LU decomposition stage.

To build and run it, navigate to its directory and use CMake:
```bash
cd extras/gemm-benchmark
mkdir build
cd build
cmake ..
cmake --build .
./gemm_benchmark
```