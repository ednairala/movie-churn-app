# Start from a minimal official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements FIRST (from its real location) to leverage Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy production source (includes model.pkl, so no cold-start retrain in-container)
COPY app/ ./app/

# Expose FastAPI's port
EXPOSE 8000

# Serve via Uvicorn on all interfaces
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
