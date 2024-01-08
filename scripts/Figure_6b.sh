#!/bin/bash

declare -a graph_name=("LiveJournal" "Orkut" "Friendster" "Twitter")

# 1) INFINEL with real-world datasets (1, 1, 19, 139 sec)
for i in "${graph_name[@]}"
do
  ./tl-infinel $i /var/INFINEL/dataset 78000 13335000 1200 n n n n
done

# 2) INFINEL-SD with real-world datasets (1, 1, 15, 97 sec)
for i in "${graph_name[@]}"
do
  ./tl-infinel $i /var/INFINEL/dataset 78000 13335000 1200 y y n n
done
