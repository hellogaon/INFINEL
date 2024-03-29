# INFINEL: An efficient GPU-based processing method for unpredictable large output graph queries [PPoPP'24]

This repository contains code for our paper INFINEL, an efficient GPU-based processing method for unpredictable large output graph queries. We conducted several experiments using INFINEL on the triangle listing query, which is a representative query for unpredictable large output queries. This repository provides most of the experiments performed in the paper.

Prerequisite
--------

Make sure you have installed all of the following prerequisites on your development machine.

### Run locally
- g++ 7.5.0
- CMake 3.10
- Boost 1.77.0
- CUDA Toolkit 11.6 or later
- Nvidia Driver 510.39.01 or higher

### Run with Docker
- Docker 24.0.2 or later
- CUDA Toolkit 11.6 or later
- Nvidia Driver 510.39.01 or higher
- NVIDIA Container Toolkit

Detailed installation guides for each component can be found [here](https://github.com/hellogaon/INFINEL/tree/main/prerequisite).


Hardware Setting
--------
The hardware specifications used in the paper are shown below.
- **OS:** Ubuntu 20.04
- **CPU:** 16-core 3.0 GHz CPU * 2
- **GPU:** A100 with a capacity of 80 GB
- **Memory:** 1 TB
- **SSD:** PCI-E SSD 6.4 TB

To run all experiments shown in the paper with the almost same parameters, an environment with 128 GB or more of main memory and 80 GB or more of GPU memory is required. If you are equipped with a GPU with less GPU memory than 80 GB, you can achieve similar results by adjusting the parameters. The installed GPU must be Compute Capability 6.0 or later, and has been verified to work with NVIDIA A100, NVIDIA TITAN V GPUs.


Getting Started Guide
--------
### Run locally

**1. Clone the source code**
```
$ git clone https://github.com/hellogaon/INFINEL.git
$ cd INFINEL
```

**2. Change `CMakeLists.txt` file**
1. Compute Capability setting ([reference](https://developer.nvidia.com/cuda-gpus))
```
# If the installed GPU has a Compute Capability of 8.0,

list(APPEND CUDA_NVCC_FLAGS 
	-gencode arch=compute_80,code=sm_80
	-O3 -std=c++14 -Xcompiler -fopenmp)
```
2) Boost library path setting
```
# If Boost library is installed in ~/local,

set(BOOST_ROOT "~/local")
set(BOOST_INCLUDEDIR "~/local/include")
set(BOOST_LIBRARYDIR "~/local/lib")
```

**3. Build**
```
$ mkdir build
$ cd build
$ cmake ../
$ make -j `nproc`
```

**4. Run sample query** ([expected output](https://github.com/hellogaon/INFINEL/blob/main/results/Sample/sample.txt))
```
$ ./tl-infinel RMAT08 ./../dataset/sample 10000 1000000 12 n n n y
```

### Run with Docker
**1. Clone the source code**
```
$ git clone https://github.com/hellogaon/INFINEL.git
$ cd INFINEL
```

**2. Change `CMakeLists.txt` file**
1. Compute Capability setting ([reference](https://developer.nvidia.com/cuda-gpus))
```
# If the installed GPU has a Compute Capability of 8.0,

list(APPEND CUDA_NVCC_FLAGS 
	-gencode arch=compute_80,code=sm_80
	-O3 -std=c++14 -Xcompiler -fopenmp)
```

**3. Build Dockerfile**
```
$ docker build --tag infinel .
```
**4. Run Docker**
```
# Change {/your/dataset/path} to your dataset storage path

$ docker run -it --name infinel \
                 --runtime=nvidia \
                 --gpus all \
                 -v /your/dataset/path:/var/INFINEL/dataset \
                 infinel
```

**5. Run sample query** ([expected output](https://github.com/hellogaon/INFINEL/blob/main/results/Sample/sample.txt))
```
$ ./tl-infinel RMAT08 /root/INFINEL/dataset/sample 10000 1000000 12 n n n y
```


Loading Datasets
--------

The datasets used in the experiments can be downloaded [here](https://figshare.com/articles/dataset/INFINEL_dataset/24584862). For real-world datasets such as [LiveJournal](https://snap.stanford.edu/data/com-LiveJournal.html), [Orkut](https://snap.stanford.edu/data/com-Orkut.html), [Friendster](https://snap.stanford.edu/data/com-Friendster.html), and [Twitter](https://anlab-kaist.github.io/traces/WWW2010) datasets, we provide preprocessed data (remove isolated vertices and remove self-loop edges) in CSR format of the original graph data. Please load `*.graph_info` and `*.graph` files without any subfolders in the dataset path. 
```
$ cd /your/dataset/path

# Download RMAT24 dataset (710 MB)
$ wget -O RMAT24.tar https://figshare.com/ndownloader/files/43187529
$ tar -xvf RMAT24.tar

# Download RMAT25 dataset (1.5 GB)
$ wget -O RMAT25.tar https://figshare.com/ndownloader/files/43824444
$ tar -xvf RMAT25.tar

# Download RMAT26 dataset (3.0 GB)
$ wget -O RMAT26.tar https://figshare.com/ndownloader/files/43187535
$ tar -xvf RMAT26.tar

# Download RMAT27 dataset (6.1 GB)
$ wget -O RMAT27.tar https://figshare.com/ndownloader/files/43823835
$ tar -xvf RMAT27.tar

# Download LiveJournal dataset (101 MB)
$ wget -O LiveJournal.tar https://figshare.com/ndownloader/files/43823820
$ tar -xvf LiveJournal.tar

# Download Orkut dataset (290 MB)
$ wget -O Orkut.tar https://figshare.com/ndownloader/files/43823823
$ tar -xvf Orkut.tar

# Download Friendster dataset (6.3 GB)
$ wget -O Friendster.tar https://figshare.com/ndownloader/files/43823979
$ tar -xvf Friendster.tar

# Download Twitter dataset (3.4 GB)
$ wget -O Twitter.tar https://figshare.com/ndownloader/files/43824720
$ tar -xvf Twitter.tar
```


Parameter Descriptions
--------
### Command
```
$ ./tl-infinel <GRAPH_NAME> <GRAPH_PATH> \
               <GPU_MEMORY_BUF_SIZE:MB> \
               <CHUNK_NUM> <CHUNK_SIZE:BYTE> \
               <LOAD_BALANCE_MODE_FLAG> <DOUBLE_BUFFERING_MODE_FLAG> \
               <ONLY_KERNEL_MODE_FLAG> <VERIFICATION_MODE_FLAG>
```

### Input graph parameter

- **GRAPH_NAME($1):** Graph name. `e.g., RMAT24`
- **GRAPH_PATH($2):** Dataset path. `e.g., /var/INFINEL/dataset`

### GPU memory parameter

- **GPU_MEMORY_BUF_SIZE:MB($3):** The total GPU memory buffer size used to perform the query including the GPU output buffer. `e.g., 78000`

### CAT method parameter (Section 3 in the paper)

- **CHUNK_NUM($4):** Number of chunks in the CAT method. `e.g., 13335000`
- **CHUNK_SIZE:BYTE($5):** Size of chunk in CAT method. `e.g., 1200`

### Optimization mode parameter (Section 4 in the paper)

- **LOAD_BALANCE_MODE_FLAG($6):** Whether to use Thread block segmentation mode. `(y/n)`
- **DOUBLE_BUFFERING_MODE_FLAG($7):** Whether to use Double buffering mode. `(y/n)`

### Other execution mode parameter

- **ONLY_KERNEL_MODE_FLAG($8):** Whether to use mode to get kernel execution time. `(y/n)`
- **VERIFICATION_MODE_FLAG($9):** Whether to use validation and full triangle list output mode. `(y/n)`

Parameters Setting Guide
--------

**GPU_MEMORY_BUF_SIZE:MB($3):** Set not to exceed the GPU memory of the equipped GPU. We recommend maximizing the GPU memory but setting the parameters with 1-2 GB of free space. `e.g., 78000 (for A100 with 80 GB GPU memory)`


**CHUNK_NUM($4), CHUNK_SIZE($5):** Use `CHUNK_NUM * CHUNK_SIZE` bytes as an GPU output buffer in GPU memory. The size of the GPU output buffer can be allocated within **GPU_MEMORY_BUF_SIZE($3)** minus the size of the memory used to perform the query. For example, for RMAT24, can use approximately **GPU_MEMORY_BUF_SIZE($3)** - 5,000 MB as the maximum GPU output buffer size, and for RMAT26, can use approximately **GPU_MEMORY_BUF_SIZE($3)** - 9,000 MB as the maximum GPU output buffer size. If you want to run the RMAT26 dataset with **GPU_MEMORY_BUF_SIZE($3)** set to `22000`, the GPU output buffer size cannot be larger than 13 GB.

**CHUNK_NUM($4):** It is necessary to be set to at least the number of GPU kernel threads.

**CHUNK_SIZE($5):** To store one triangle, we need 12 bytes (4 bytes * 3). Therefore, it is necessary to set it to a multiple of 12. (e.g., `1200`)


Evaluation and Expected Results
--------

We present how to reproduce the experimental results presented in the paper. You can experiment with all the queries used in the paper. The following query was experimented on an A100 GPU with 80 GB of GPU memory. Please note that the **GPU_MEMORY_BUF_SIZE($3)**, **CHUNK_NUM($4)**, and **CHUNK_SIZE($5)** parameters must be changed depending on your installed GPU. The time in parentheses indicates the execution time for each query in our experimental environment.

### INFINEL query execution time with synthetic datasets (Figure 6a, [expected output](https://github.com/hellogaon/INFINEL/tree/main/results/Figure%206a))
```
# GPU output buffer size: 16 GB
# Chunk size: 1.2 KB
# Total output size: 142.5 GB, 275.5 GB, 776.2 GB, 2.0 TB

# 1) INFINEL with synthetic datasets (19, 40, 112, 301 sec)
$ ./tl-infinel RMAT24 /var/INFINEL/dataset 78000 13335000 1200 n n n n
$ ./tl-infinel RMAT25 /var/INFINEL/dataset 78000 13335000 1200 n n n n
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 n n n n
$ ./tl-infinel RMAT27 /var/INFINEL/dataset 78000 13335000 1200 n n n n

# 2) INFINEL-SD with synthetic datasets (12, 24, 65, 170 sec)
$ ./tl-infinel RMAT24 /var/INFINEL/dataset 78000 13335000 1200 y y n n
$ ./tl-infinel RMAT25 /var/INFINEL/dataset 78000 13335000 1200 y y n n
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 y y n n
$ ./tl-infinel RMAT27 /var/INFINEL/dataset 78000 13335000 1200 y y n n
```

### INFINEL query execution time with real-world datasets (Figure 6b, [expected output](https://github.com/hellogaon/INFINEL/tree/main/results/Figure%206b))
```
# GPU output buffer size: 16 GB
# Chunk size: 1.2 KB
# Total output size: 3.4 GB, 7.5 GB, 50.1 GB, 417.9 GB

# 1) INFINEL with real-world datasets (1, 1, 19, 139 sec)
$ ./tl-infinel LiveJournal /var/INFINEL/dataset 78000 13335000 1200 n n n n
$ ./tl-infinel Orkut /var/INFINEL/dataset 78000 13335000 1200 n n n n
$ ./tl-infinel Friendster /var/INFINEL/dataset 78000 13335000 1200 n n n n
$ ./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 n n n n

# 2) INFINEL-SD with real-world datasets (1, 1, 15, 97 sec)
$ ./tl-infinel LiveJournal /var/INFINEL/dataset 78000 13335000 1200 y y n n
$ ./tl-infinel Orkut /var/INFINEL/dataset 78000 13335000 1200 y y n n
$ ./tl-infinel Friendster /var/INFINEL/dataset 78000 13335000 1200 y y n n
$ ./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 y y n n
```

###  INFINEL kernel execution time (Table 2, [expected output](https://github.com/hellogaon/INFINEL/tree/main/results/Table%202))
```
# GPU output buffer size: 16 GB
# Chunk size: 1.2 KB
# Total output size: 3.4 GB ~ 2 TB

# 1) INFINEL(K) with synthetic datasets (7.3, 17.9, 50.4, 145.5 sec)
$ ./tl-infinel RMAT24 /var/INFINEL/dataset 78000 13335000 1200 n n y n
$ ./tl-infinel RMAT25 /var/INFINEL/dataset 78000 13335000 1200 n n y n
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 n n y n
$ ./tl-infinel RMAT27 /var/INFINEL/dataset 78000 13335000 1200 n n y n

# 2) INFINEL(K) with real-world datasets (0.3, 0.7, 15.0, 106.7 sec)
$ ./tl-infinel LiveJournal /var/INFINEL/dataset 78000 13335000 1200 n n y n
$ ./tl-infinel Orkut /var/INFINEL/dataset 78000 13335000 1200 n n y n
$ ./tl-infinel Friendster /var/INFINEL/dataset 78000 13335000 1200 n n y n
$ ./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 n n y n
```

###  INFINEL kernel execution time varying output buffer size (Figure 7a, [expected output](https://github.com/hellogaon/INFINEL/tree/main/results/Figure%207))
```
# GPU output buffer size: 64 GB, 32 GB, 16 GB, 8 GB, 4 GB
# Chunk size: 1.2 KB
# Total output size: 776.2 GB

# 1) INFINEL(K) with output buffer size of 64 GB (50 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 53335000 1200 n n y n
# 2) INFINEL(K) with output buffer size of 32 GB (50 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 26667500 1200 n n y n
# 3) INFINEL(K) with output buffer size of 16 GB (50 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 n n y n
# 4) INFINEL(K) with output buffer size of 8 GB (51 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 6667500 1200 n n y n
# 5) INFINEL(K) with output buffer size of 4 GB (52 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 3335000 1200 n n y n
```

###  INFINEL query execution time varying output buffer size (Figure 8a, [expected output](https://github.com/hellogaon/INFINEL/tree/main/results/Figure%208a))
```
# GPU output buffer size: 32 GB, 16 GB, 8 GB, 4 GB, 2 GB, 1 GB
# Chunk size: 1.2 KB
# Total output size: 776.2 GB

# 1) INFINEL with output buffer size of 32 GB (111.4 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 26667500 1200 n n n n
# 2) INFINEL with output buffer size of 16 GB (111.5 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 n n n n
# 3) INFINEL with output buffer size of 8 GB (112.4 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 6667500 1200 n n n n
# 4) INFINEL with output buffer size of 4 GB (114.1 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 3335000 1200 n n n n
# 5) INFINEL with output buffer size of 2 GB (119.6 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 1667500 1200 n n n n
# 6) INFINEL with output buffer size of 1 GB (137.9 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 835000 1200 n n n n
```

###  INFINEL query execution time varying chunk size (Figure 8b, [expected output](https://github.com/hellogaon/INFINEL/tree/main/results/Figure%208b))
```
# GPU output buffer size: 16 GB
# Chunk size: 0.3 KB, 1.2 KB, 4.8 KB, 19.2 KB, 76.8 KB
# Total output size: 776.2 GB

# 1) INFINEL with chunk size of 0.3 KB (112.7 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 53340000 300 n n n n
# 2) INFINEL with chunk size of 1.2 KB (111.5 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 n n n n
# 3) INFINEL with chunk size of 4.8 KB (112.7 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 3333750 4800 n n n n
# 4) INFINEL with chunk size of 19.2 KB (120.0 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 833438 19200 n n n n
# 5) INFINEL with chunk size of 76.8 KB (176.4 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 208360 76800 n n n n
```

###  INFINEL performance breakdown with synthetic dataset (Figure 9a, [expected output](https://github.com/hellogaon/INFINEL/tree/main/results/Figure%209a))
```
# GPU output buffer size: 16 GB
# Chunk size: 1.2 KB
# Total output size: 776.2 GB
# Idea #1: Chunk allocation per thread and kernel context for stop and restart (Section 3)
# Idea #2: Thread block segmentation (Section 4.1)
# Idea #3: Double buffering (Section 4.2)

# 1) Use Idea #1 with RMAT26 dataset (112 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 n n n n
# 2) Use Idea #1 and #2 with RMAT26 dataset (107 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 y n n n
# 3) Use Idea #1 and #3 with RMAT26 dataset (70 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 n y n n
# 4) Use Idea #1, #2 and #3 with RMAT26 dataset (65 sec)
$ ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 y y n n
```

###  INFINEL performance breakdown with real-world dataset (Figure 9b, [expected output](https://github.com/hellogaon/INFINEL/tree/main/results/Figure%209b))
```
# GPU output buffer size: 16 GB
# Chunk size: 1.2 KB
# Total output size: 417.9 GB
# Idea #1: Chunk allocation per thread and kernel context for stop and restart (Section 3)
# Idea #2: Thread block segmentation (Section 4.1)
# Idea #3: Double buffering (Section 4.2)

# 1) Use Idea #1 with Twitter dataset (139 sec)
$ ./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 n n n n
# 2) Use Idea #1 and #2 with Twitter dataset (128 sec)
$ ./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 y n n n
# 3) Use Idea #1 and #3 with Twitter dataset (108 sec)
$ ./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 n y n n
# 4) Use Idea #1, #2 and #3 with Twitter dataset (97 sec)
$ ./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 y y n n
```

Experiment Customization
--------

### Change GPU output buffer size

You can perform the experiment by adjusting the **CHUNK_NUM($4)** and **CHUNK_SIZE($5)** parameters.

### Change the number of threads

You can experiment by modifying the number of thread blocks (`MaxBlocks`) and the number of threads per thread block (`NumThread`)  in `source/query/tl-infinel/tl_defines.h`.



Claims from the paper supported by the artifact
--------
This artifact allows us to perform most of the experiments shown in the paper, and supports all the claims in the paper. Representatively, we show that INFINEL can handle unpredictable large output graph queries in a one-phase method without a pre-compute phase, and significantly outperforms conventional two-phase methods. We also demonstrated that INFINEL maintains consistent performance without a significant impact on the GPU output buffer size. Finally, as INFINEL has been distributed as a library, it should be easy to adapt it to other large output queries.


Note
--------

On some CPUs, it is sometimes not possible to allocate several gigabytes of pinned memory, even though there is sufficient free space in the main memory. This can be resolved by turning off the CPU's IOMMU in the BIOS. ([link](https://forums.developer.nvidia.com/t/kernel-call-trace-observed-when-calling-cudafreehost-cudahostalloc-for-buffers-on-amd-cpu-with-nvi/72930/3))
