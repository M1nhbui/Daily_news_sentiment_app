# Use a slim Python base
FROM python:3.10-slim

# Install system-level dependencies for psycopg2 and scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy everything into container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run the dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
