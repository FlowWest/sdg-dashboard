FROM python:3.13.0 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY --from=flyio/litefs:0.5 /usr/local/bin/litefs /usr/local/bin/litefs

RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt
RUN .venv/bin/pip install -i https://test.pypi.org/simple/ hecdss
FROM python:3.13.0-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "fly-local-6pn", "--port", "8000"]
