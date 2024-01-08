#!/bin/bash

# INFINEL query execution time varying output buffer size (Figure 8a)
declare -a chunk_num=("26667500" "13335000" "6667500" "3335000" "1667500" "835000")

for i in "${chunk_num[@]}"
do
  ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 $i 1200 n n n n
done