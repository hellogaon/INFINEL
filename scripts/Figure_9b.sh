#!/bin/bash

# INFINEL performance breakdown with real-world dataset (Figure 9b)
./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 n n n n
./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 y n n n
./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 n y n n
./tl-infinel Twitter /var/INFINEL/dataset 78000 13335000 1200 y y n n