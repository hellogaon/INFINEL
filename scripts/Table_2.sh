#!/bin/bash

# INFINEL kernel execution time (Table 2)
declare -a graph_name=("RMAT24" "RMAT25" "RMAT26" "RMAT27" "LiveJournal" "Orkut" "Friendster" "Twitter")

for i in "${graph_name[@]}"
do
  ./tl-infinel $i /var/INFINEL/dataset 78000 13335000 1200 n n y n
done