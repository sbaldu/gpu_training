# Introduction to GPU Programming 
### Presentation:
You can find the presentation here:
<a href="
https://docs.google.com/presentation/d/1Id7dLpzC0UlsJCJeXdJBvwigQ1aFaDmk0l0Pgkv7bwo/edit?usp=sharing" target="_blank">
https://docs.google.com/presentation/d/1Id7dLpzC0UlsJCJeXdJBvwigQ1aFaDmk0l0Pgkv7bwo/edit?usp=sharing</a>

### Hands-on

The CUDA Runtime API reference manual is a very useful source of information:
<a href="http://docs.nvidia.com/cuda/cuda-runtime-api/index.html" target="_blank">http://docs.nvidia.com/cuda/cuda-runtime-api/index.html</a>


Check that your environment is correctly configured to compile CUDA code by running:
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```

```bash
git clone https://github.com/felicepantaleo/gpu_training.git
```

Compile and run the `deviceQuery` application:
```bash
$ cd hands-on/utils/deviceQuery
$ make
```

You can get some useful information about the features and the limits that you will find on the device you will be running your code on. For example:

```shell
$ ./deviceQuery
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Tesla V100-SXM2-32GB"
  CUDA Driver Version / Runtime Version          12.9 / 12.9
  CUDA Capability Major/Minor version number:    7.0
  Total amount of global memory:                 32494 MBytes (34072559616 bytes)
  (80) Multiprocessors, ( 64) CUDA Cores/MP:     5120 CUDA Cores
  GPU Max Clock rate:                            1530 MHz (1.53 GHz)
  Memory Clock rate:                             877 Mhz
  Memory Bus Width:                              4096-bit
  L2 Cache Size:                                 6291456 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        98304 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 5
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.9, CUDA Runtime Version = 12.9, NumDevs = 1
Result = PASS
```

Some of you are sharing the same machine and some time measurements can be influenced by other users running at the very same moment. It can be necessary to run time measurements multiple times.


# GPU Training — Hands‑On Exercises

These short, self‑contained labs walk students from CUDA basics to a tiny predator–prey simulation.  Every exercise follows the same workflow:

```
1. Fill in all  ►►► TODO ◄◄◄  and/or kernel bodies.
2. Build with the line shown in the banner comment.
3. Run → the program prints “…PASSED 🎉” when assertions succeed.
```

All code lives under `gpu_training/` and compiles on CUDA 12 + gcc 13 (Ubuntu 22.04).

---

## Exercise 1 – CUDA Memory Model

*Goal – feel the separation between CPU (host) and GPU (device) address spaces.*

| Step | What you implement                                   |
| ---- | ---------------------------------------------------- |
|  1   | `cudaMallocAsync` two device buffers `d_a` and `d_b` |
|  2   | Copy host array `h_a → d_a`                          |
|  3   | Copy device array `d_a → d_b` (device‑to‑device)     |
|  4   | Copy back `d_b → h_a`                                |
|  5   | Free `d_a`, `d_b` with `cudaFreeAsync`               |

```bash
nvcc -std=c++20 memory_model.cu -o ex01
./ex01   # prints “Exercise 1 – memory model: PASSED 🎉”
```

### Variation

Add a non‑blocking version using **streams** + `cudaMemcpyAsync` and time a 100 MB H↔D copy to estimate PCIe bandwidth.

---

## Exercise 2 – Launch Your First Kernel

*Goal – understand grid/block configuration and indexing.*

1. `cudaMallocAsync` a device array `d_a[N]`
2. Launch a **1‑D grid** of **1‑D blocks**

   ```c++
   d_a[i] = i + 42;  // each thread writes one element
   ```
3. Copy back, verify, free the memory.

Compile & run:

```bash
nvcc -std=c++20 launch_kernel.cu -o ex02 && ./ex02
```

**Hint:** Global thread index = `blockIdx.x * blockDim.x + threadIdx.x`.

---

## Exercise 3 – 2‑D Grid & Block

*Goal – move from 1‑D to 2‑D indexing.*

Matrix **M\[numRows × numCols]**.

| Task | Detail                                                                                          |
| ---- | ----------------------------------------------------------------------------------------------- |
| 1    | Set `numRows`, `numCols` (start with 4 × 4, then 19 × 67).                                      |
| 2    | Launch a **2‑D grid of 2‑D blocks** so each thread writes<br>`M[row,col] = row * numCols + col` |
| 3    | Copy to host, assert correctness                                                                |
| 4    | Experiment: fix block = `16×16`, compute `blocksPerGrid` with ceiling division                  |

Compile:

```bash
nvcc -std=c++20 ex03_fill_matrix.cu -o ex03 && ./ex03
```

---

## Exercise 4 – Parallel Reduction ∑

*Goal – sum a 1‑D array faster than the CPU.*

> **Rule of thumb:** keep each block power‑of‑two and reduce in shared memory.

1. Kernel #1: each block loads its slice, does **shared‑mem tree reduction**, writes one partial sum.
2. Kernel #2: single block reduces those partials into the final total.
3. Copy result, compare to `std::accumulate`.

*Bonus*: one‑step reduction (single kernel).

---

## Thrust

The four labs below mirror the CUDA ones but use the Thrust C++ library. All code lives in hands-on/thrust/XX_<name>/.
Every exercise follows the same workflow:

1. Open the `.cu` file and replace each ►►► TODO ◄◄◄.
2. Build with the provided Makefile:   `$ make`
3. Run → the program prints “…PASSED 🎉” (or an assert succeeds).

### Exercise 1 – Memory Model with Thrust Vectors

1	Initialize a `thrust::host_vector<int> h_buf(dim)` with 0, 1, 2, … (hint : `thrust::sequence`)
2	Create two `thrust::device_vector<int>` buffers `d1`, `d2`.
3	Copy host → d1 with any Thrust copy method.
4	Copy d1 → d2 using a different method (e.g. explicit iterator pair).
5	Zero h_buf on the CPU.
6	Copy d2 → h_buf and assert that values are again 0…dim-1.

### Exercise 2 – Your First Thrust “Kernel”

    Pinned-host buffer h_a (use thrust::universal_vector or cudaMallocHost).

    Device buffer d_a (thrust::device_vector<int>(N)).

    Fill d_a with i + 42 using thrust::sequence.

    Create a cudaStream_t queue.

    Asynchronously copy d_a → h_a with thrust::copy(thrust::cuda::par.on(queue), …).

    Verify with an assert that h_a == {42, 43, …}.

### Exercise 3 – Data Statistics

*Folder *: thrust/03_data_statistics
*Goal *: combine transform and reduce to compute simple stats.

1.	Host vector `h_data(N)`; fill with random floats (provided helper).
2.	Copy to `thrust::device_vector<float> d_data`.
3.	`Mean = thrust::reduce(d_data.begin(), d_data.end()) / N`.
4.	`squared_diffs[i] = (d_data[i] – mean)²` via `thrust::transform`.
5.	`Stdev = sqrt(reduce(squared_diffs)/N)`.
6.	Challenge: recompute stdev without the intermediate buffer.

### Exercise 4 – Maximum Absolute Difference (Zip Iterators)

Two small host vectors `h_v1`, `h_v2` already provided.

Copy to `d_v1`, `d_v2`.

Use `thrust::transform` with a zip iterator pair to compute.
