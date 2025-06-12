# Use Python 3.8 base image
FROM python:3.8-slim

# Set environment variables to disable Python cache
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/static/images

# Copy only what's needed for model.py to run
COPY model.py .
COPY data/ThoracicCancerSurgery.csv data/

# Run model.py once to train and save models, graphs, etc.
RUN python model.py

# Copy the rest of the application
COPY . .

# Expose Flask app port
EXPOSE 5000

# Default command to run Flask app
CMD ["python", "app.py"]
