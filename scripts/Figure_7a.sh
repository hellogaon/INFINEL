#!/bin/bash

# INFINEL kernel execution time varying output buffer size (Figure 7a)
declare -a chunk_num=("53335000" "26667500" "13335000" "6667500" "3335000")

for i in "${chunk_num[@]}"
do
  ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 $i 1200 n n y n
done