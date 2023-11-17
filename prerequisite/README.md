# Prerequisite Install Guide

This document describes how to install the prerequisites to run INFINEL.

### Run locally

- g++ 7.5.0
```
$ sudo apt-get install -y gcc-7 g++-7 build-essential
```

- Cmake 3.10 ([link](https://cmake.org/files/v3.10))
```
$ wget https://cmake.org/files/v3.10/cmake-3.10.0-Linux-x86_64.tar.gz
$ tar xvfz cmake-3.10.0-Linux-x86_64.tar.gz
$ export CMAKE_HOME=/your/path/cmake-3.10.0-Linux-x86_64
$ export PATH=$CMAKE_HOME/bin:$PATH
```

- Boost 1.77.0 ([link](https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/))
```
$ wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz
$ tar xvfz boost_1_77_0.tar.gz
$ cd boost_1_77_0
$ sudo ./bootstrap.sh --prefix=$HOME/local
$ sudo ./b2 install
```

- CUDA Toolkit 11.6 and Nvidia Driver 510.39.01 ([link](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local))
```
$ wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
$ sudo sh cuda_11.6.0_510.39.01_linux.run
```

### Run with docker

- Docker latest version ([link](https://docs.docker.com/engine/install/ubuntu/))
```
$ sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

-  CUDA Toolkit 11.6 and Nvidia Driver 510.39.01 ([link](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local))
```
$ wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
$ sudo sh cuda_11.6.0_510.39.01_linux.run
```

- NVIDIA Container Toolkit ([link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
```
$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update
$ sudo apt-get install -y nvidia-container-toolkit
$ sudo nvidia-ctk runtime configure --runtime=docker
$ sudo systemctl restart docker
```

