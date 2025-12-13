# Use NVIDIA's PyTorch image with CUDA 12.1 (VALID TAG)
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Install OpenCV and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Expose port
EXPOSE 8000

# Start app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# docker run --gpus all -d   -p 8000:8000   -v "$USERPROFILE/.aws:/root/.aws"   --name people_container   people_image 