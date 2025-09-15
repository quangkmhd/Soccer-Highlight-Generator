#!/bin/bash

# Stop and remove the existing container if it exists
if [ "$(docker ps -aq -f name=soccer-highlight)" ]; then
    docker stop soccer-highlight
    docker rm soccer-highlight
fi

# Run the new container
docker run -d --name soccer-highlight \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    -p 8000:8000 \
    -p 7860:7860 \
    -v "$(pwd)/input_video:/app/input_video" \
    -v "$(pwd)/pipeline_output:/app/pipeline_output" \
    -v "$(pwd)/weight:/app/weight" \
    -v "$(pwd)/app_v2/uploads:/app/app_v2/uploads" \
    --restart unless-stopped \
    soccer-highlight:latest

echo "Container soccer-highlight is starting."
