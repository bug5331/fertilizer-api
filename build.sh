#!/bin/bash

# Create model directory
mkdir -p saved_model

# Download model file using gdown
gdown --id 1HqQVeFIYst7xGidIDEfHcXyLb8W6B_Yg -O saved_model/final_model.h5

