FROM python:3.11-slim

WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies using the pinned versions
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]