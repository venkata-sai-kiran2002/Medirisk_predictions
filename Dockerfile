FROM python:3.8-slim

WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p static/images

# Copy application files
COPY . .

# Set permissions
RUN chmod -R 755 .

CMD ["python", "app.py"]