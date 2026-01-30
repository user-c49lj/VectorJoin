#!/bin/bash
# Usage: ./build_nsg.sh <fvecs> <graph_out> <nsg_out> <num_rows> <ood_txt> <ood_flag>

INPUT_FVECS=$1
GRAPH_OUT=$2
NSG_OUT=$3
NUM_ROWS=$4
OOD_TXT=$5
OOD_FLAG=$6

# 1. Build the NN-Descent Graph
./efanna_graph/tests/test_nndescent $INPUT_FVECS $GRAPH_OUT 400 400 12 15 100   # only for nytimes change 15 to 50 in this line

# 2. Build the NSG Index with your specific parameters
# Parameters: input_fvecs, input_graph, L, R, C, num_rows, 0.5, out_nsg, ood_txt, ood_flag
./nsg/build/tests/test_nsg_index $INPUT_FVECS $GRAPH_OUT 60 70 500 $NUM_ROWS 0.5 $NSG_OUT $OOD_TXT $OOD_FLAG