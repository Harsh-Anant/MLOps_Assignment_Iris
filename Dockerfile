# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
#COPY app.py .
#COPY mlruns ./mlruns

COPY . /app

RUN test -f app.py

# Set the MLFLOW_TRACKING_URI environment variable inside the container
ENV MLFLOW_TRACKING_URI="file:///app/mlruns"

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
