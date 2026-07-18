#!/bin/bash

# Activate the virtual environment if needed
source ../venv/bin/activate || true

# Find all chunk files, sort them alphabetically to process sequentially
for chunk in $(ls eval_chunk_* | sort); do
    echo "=========================================================="
    echo "Starting training on chunk: $chunk"
    echo "=========================================================="
    
    # Run the PyTorch training
    python3 train_halfkp.py "$chunk"
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo "Successfully trained on $chunk. Deleting file to save space..."
        rm "$chunk"
        echo "Deleted $chunk."
    else
        echo "Error occurred during training on $chunk. Stopping automation."
        exit 1
    fi
done

echo "🎉 All chunks have been processed! The massive 101GB dataset is complete!"
