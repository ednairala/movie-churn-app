# Start from a minimal official Python image
FROM python:3.11-slim

# Set the working directory inside the container context
WORKDIR /app

# Install native system-level dependencies required for heavy data frameworks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements configuration first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install all dependencies directly from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your restructured production app source files into the image layer
COPY app/ ./app/

# Copy pre-trained model if it exists (prevents cold-start rebuild inside container)
COPY app/model.pkl ./app/model.pkl

# Expose FastAPI's internal network interface port
EXPOSE 8000

# Serve application via Uvicorn bound to all network interfaces
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]