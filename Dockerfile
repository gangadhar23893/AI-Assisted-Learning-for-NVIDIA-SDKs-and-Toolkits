# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for faiss and sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "components/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

