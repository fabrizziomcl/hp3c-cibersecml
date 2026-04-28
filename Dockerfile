FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure logs and models directories exist
RUN mkdir -p logs models data/raw data/processed data/external dataset

# Environment variable for PYTHONPATH
ENV PYTHONPATH=/app

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
