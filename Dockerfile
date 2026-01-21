FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libgtk-3-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python && \
    pip install opencv-contrib-python-headless

# Copy application code
COPY backend/ .

# Download YOLOv8 model if not present (just in case it's needed by other modules)
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt', 'yolov8n.pt')" || echo "Model download failed, will be downloaded at runtime"

# Expose port
EXPOSE 10000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/health').read()" || exit 1

# Set environment variables
ENV PORT=10000
ENV WS_HOST=0.0.0.0

# Run the application
CMD ["python", "movements.py"]