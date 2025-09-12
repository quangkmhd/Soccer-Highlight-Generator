FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    && update-alternatives --install /usr/local/bin/python python /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install main project dependencies first
RUN pip install --no-cache-dir -r requirements.txt


# Create necessary directories
RUN mkdir -p app_v2/uploads \
    && mkdir -p pipeline_output \
    && mkdir -p weight/action \
    && mkdir -p weight/ball \
    && mkdir -p weight/camera

# Expose ports for FastAPI and Gradio
EXPOSE 8000 7860

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    cd /app && python3 -m app_v2.main_api\n\
elif [ "$1" = "gradio" ]; then\n\
    cd /app && python3 -m app_v2.gradio_app\n\
elif [ "$1" = "both" ]; then\n\
    cd /app && python3 -m app_v2.main_api &\n\
    cd /app && python3 -m app_v2.gradio_app &\n\
    wait\n\
else\n\
    cd /app && python3 -m app_v2.main_api\n\
fi' > /entrypoint.sh && chmod +x /entrypoint.sh

# Default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["both"]
