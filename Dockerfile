# ── ReWaterGAP Web Interface ──────────────────────────────────
# Build:  docker compose build
# Run:    docker compose up
# ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System deps needed by netCDF4 / HDF5 / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev libnetcdf-dev gcc g++ \
        libfreetype6-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (pinned from requirements.txt + web deps)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

# Copy application code (input_data is excluded via .dockerignore)
COPY . .

# Create directories the app expects
RUN mkdir -p jobs uploads output_data

# The input_data directory is mounted at runtime via docker-compose
# (see docker-compose.yml → volumes)

EXPOSE 8000

# Use production-grade uvicorn (no --reload in production)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
