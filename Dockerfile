# Base image with PyTorch and CUDA support pre-installed
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies for C++ compilation and GDAL (rasterio)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    gdal-bin \
    libgdal-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

# Copy the entire project
COPY . .

# Install JupyterLab for interactive data science
RUN pip install jupyterlab

# Install the package in editable mode (compiles the C++ extensions)
RUN pip install -e .

# Expose Jupyter port for interactive sessions
EXPOSE 8888

# Default command: start a Jupyter server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
