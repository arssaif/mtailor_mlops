# Use a slim Python base; Cerebrium will bind GPU drivers at runtime
FROM python:3.10-slim

# Install system deps (dumb-init for proper signal handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
        dumb-init \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining files
COPY . .

# Expose the port that cerebrium.toml references
EXPOSE 8192

# Use dumb-init so Uvicorn can receive and forward signals properly
ENTRYPOINT ["dumb-init", "--"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8192"]
