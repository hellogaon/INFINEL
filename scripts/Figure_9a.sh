#!/bin/bash

# INFINEL performance breakdown with synthetic dataset (Figure 9a)
./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 n n n n
./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 y n n n
./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 n y n n
./tl-infinel RMAT26 /var/INFINEL/dataset 78000 13335000 1200 y y n n