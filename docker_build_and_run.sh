#!/bin/bash

# Build the Docker image
docker build -t experimenteer:latest .

# Run the Docker container
docker run -p 8501:8501 experimenteer:latest