#!/bin/bash

declare -a graph_name=("RMAT24" "RMAT25" "RMAT26" "RMAT27")

# INFINEL query execution time with synthetic datasets (Figure 6a)
# 1) INFINEL with synthetic datasets
for i in "${graph_name[@]}"
do
  ./tl-infinel $i /var/INFINEL/dataset 78000 13335000 1200 n n n n
done

# 2) INFINEL-SD with synthetic datasets
for i in "${graph_name[@]}"
do
  ./tl-infinel $i /var/INFINEL/dataset 78000 13335000 1200 y y n n
done
