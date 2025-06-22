# Gunakan image Python resmi
FROM python:3.10-slim

# Set direktori kerja di container
WORKDIR /app

# Copy file proyek ke container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable (opsional)
ENV PYTHONUNBUFFERED=1

# Jalankan app Flask
CMD ["python", "app.py"]
