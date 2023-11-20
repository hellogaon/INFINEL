FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt install -y wget git

# install g++
RUN apt-get install -y gcc-7 g++-7 build-essential

WORKDIR /root/Util

# install cmake
RUN wget https://cmake.org/files/v3.10/cmake-3.10.0-Linux-x86_64.tar.gz
RUN tar xvfz cmake-3.10.0-Linux-x86_64.tar.gz
ENV CMAKE_HOME=/root/Util/cmake-3.10.0-Linux-x86_64
ENV PATH=$CMAKE_HOME/bin:$PATH

# install boost
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz
RUN tar xvfz boost_1_77_0.tar.gz
WORKDIR /root/Util/boost_1_77_0
RUN ./bootstrap.sh --prefix=/root/local
RUN ./b2 install
ENV LD_LIBRARY_PATH=/root/local/lib:$LD_LIBRARY_PATH

# build INFINEL
WORKDIR /root
RUN git clone https://github.com/hellogaon/INFINEL.git
WORKDIR /root/INFINEL
RUN mkdir build
WORKDIR /root/INFINEL/build
RUN cmake ../
RUN make -j `nproc`
