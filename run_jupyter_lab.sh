#!/bin/bash

# Navigate to the directory where this script resides (project root)
cd "$(dirname "$0")"

# Start Jupyter Lab using the training virtual environment
echo "Starting Jupyter Lab..."
./.venv-train/Scripts/jupyter-lab
