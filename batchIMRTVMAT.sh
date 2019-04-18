#!/bin/bash

declare -a arr=("brain360" "lung360" "spine360")
thresholds=(0.0 0.05 0.075 0.9 0.1 0.11 0.12 0.13 0.14 0.15 0.17 0.19 0.2 0.22 0.23 0.25 0.3 0.35 0.4 0.45 0.5)
for case in "${arr[@]}"
    do
	echo "$case"
	for i in "${thresholds[@]}"
	do
	    echo $i
	    /opt/intel/intelpython3/bin/python3 IMRTwithLimits.py "$case" "$i"
	done
    done
    
