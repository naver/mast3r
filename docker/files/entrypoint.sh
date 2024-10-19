#!/bin/bash

# Exit immediately if a command exits with a non-zero status, 
# print each command before executing (for debugging), and treat unset variables as an error.
set -eux

# Set default values for DEVICE and MODEL if they are not provided
DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-MASt3R_ViTLarge_BaseDecoder_512_dpt.pth}

# Log the device and model being used
echo "Running MASt3R demo with:"
echo "Model: $MODEL"
echo "Device: $DEVICE"

# Check if the model file exists in the checkpoints directory
if [ ! -f "checkpoints/$MODEL" ]; then
    echo "Error: Model file 'checkpoints/$MODEL' does not exist."
    exit 1
fi

# Execute the Python script with provided arguments
exec python3 demo.py --weights "checkpoints/$MODEL" --device "$DEVICE" --local_network "$@"
