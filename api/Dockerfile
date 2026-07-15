FROM python:3.12-slim

WORKDIR /app

# system deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# install API-only requirements (no torch)
COPY api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt

# install the engine package (editable installs don't work in Docker; copy source)
COPY engine /app/engine
RUN pip install --no-cache-dir --no-deps -e /app/engine

# copy the API app
COPY api /app/api

ENV PYTHONPATH=/app/engine
ENV PORT=8000

CMD ["sh", "-c", "uvicorn api.app.main:app --host 0.0.0.0 --port ${PORT}"]
