#!/bin/bash

# Navigate to the directory where this script resides (project root)
cd "$(dirname "$0")"

# Start the MLflow UI pointing to the SQLite database
echo "Starting MLflow UI (Local Edition)..."
./.venv-train/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db
