FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with pip cache optimization
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (only what we need)
COPY api/ ./api/
COPY public/ ./public/

# Run FastAPI
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "3000"]
