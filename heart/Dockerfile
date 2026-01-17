FROM python:3.11-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire heart folder as a package
COPY . ./heart

# Set Python path
ENV PYTHONPATH=/app

# Run from the heart module
CMD ["python", "-m", "heart.main"]