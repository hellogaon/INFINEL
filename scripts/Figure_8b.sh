#!/bin/bash

# INFINEL query execution time varying chunk size (Figure 8b)
declare -a chunk_num=("53340000" "13335000" "3333750" "833438" "208360")
declare -a chunk_size=("300" "1200" "4800" "19200" "76800")

length=${#chunk_num[@]}

for (( i=0; i<${length}; i++ ));
do
  ./tl-infinel RMAT26 /var/INFINEL/dataset 78000 ${chunk_num[$i]} ${chunk_size[$i]} n n n n
done