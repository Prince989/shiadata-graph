FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY config ./config
COPY main.py ./
COPY scripts ./scripts
COPY docker/run-loop.sh /app/docker/run-loop.sh

RUN pip install --no-cache-dir -e . \
    && sed -i 's/\r$//' /app/docker/run-loop.sh \
    && chmod +x /app/docker/run-loop.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["sh", "/app/docker/run-loop.sh"]
CMD ["python", "main.py", "--help"]
