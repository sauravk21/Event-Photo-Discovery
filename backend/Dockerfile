FROM python:3.12-slim

# build-essential: insightface is published as a source dist with a C++ extension.
# libgl1 / libglib2.0-0: runtime libraries opencv (pulled in by insightface) needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY static static

ENV DATA_DIR=/data
EXPOSE 8000
# 0.0.0.0, not the default 127.0.0.1, or the published port can't reach the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
