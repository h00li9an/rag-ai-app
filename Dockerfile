FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (curl/procps for healthchecks)
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /secureiqlab

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies globally (No venv needed inside container)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port 8001 (Matches uvicorn defaults in app.py)
EXPOSE 8001

# Run the application
ENTRYPOINT ["python", "app.py"]
