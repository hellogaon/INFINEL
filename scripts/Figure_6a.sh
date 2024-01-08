#!/bin/bash

declare -a graph_name=("RMAT24" "RMAT25" "RMAT26" "RMAT27")

# 1) INFINEL with synthetic datasets (19, 40, 112, 301 sec)
for i in "${graph_name[@]}"
do
  ./tl-infinel $i /var/INFINEL/dataset 78000 13335000 1200 n n n n
done

# 2) INFINEL-SD with synthetic datasets (12, 24, 65, 170 sec)
for i in "${graph_name[@]}"
do
  ./tl-infinel $i /var/INFINEL/dataset 78000 13335000 1200 y y n n
done
