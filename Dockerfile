FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade llama-cpp-python

# Copy only app code
COPY app ./app

EXPOSE 8000
CMD ["python","-c","print('base image ready')"]