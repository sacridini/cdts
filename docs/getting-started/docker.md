# Docker

Because CDTS utilizes C++ compilation and GPU-accelerated PyTorch for its deep learning modules, deploying it to the cloud or sharing consistent environments across research teams is most efficiently done using Docker.

## Official Docker Image

We provide a complete Dockerfile based on the official PyTorch image with CUDA support (`pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`).

The Docker image includes:
* System dependencies for C++ compilation and GDAL.
* Python dependencies and the CDTS package installed in editable mode.
* JupyterLab for interactive data science.

## Building and Running

To build the C++ engines, install PyTorch with CUDA, and launch a JupyterLab environment, you can use the provided `docker-compose.yml` file.

1. Ensure you have Docker and Docker Compose installed on your system.
2. Navigate to the root directory of the CDTS repository.
3. Run the following command:

```bash
docker-compose up --build
```

This will build the image and start a Jupyter server. By default, the server runs on port 8888. You can access it by navigating to `http://localhost:8888` in your web browser. 

The command used by the container is configured to allow root access and disables the automatic browser launch, making it ideal for headless server environments.
