# Stage 1: Build Frontend
FROM node:18-alpine as frontend_build
WORKDIR /app/frontend
COPY Frontend/package*.json ./
RUN npm ci || npm install
COPY Frontend/ .
RUN npm run build

# Stage 2: Backend Runtime
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglx-mesa0 \
    libgtk-3-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless && \
    pip install opencv-contrib-python-headless

# Verify cv2 installation
RUN python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# Copy Backend Code
COPY backend/ .

# Copy Frontend Build from Stage 1
COPY --from=frontend_build /app/frontend/dist ./static

# YOLO model download removed - not used
# RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt', 'yolov8n.pt')" || echo "Model download failed, will be downloaded at runtime"

# Expose port
EXPOSE 10000

# Healthcheck (checks /health endpoint)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/health').read()" || exit 1

# Set environment variables
ENV PORT=10000
ENV WS_HOST=0.0.0.0

# Run the application
CMD ["python", "movements.py"]