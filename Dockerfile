FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# runtime deps
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
      fastapi uvicorn joblib pandas numpy scikit-learn

COPY src ./src
COPY INFERENCE_CONTRACT.md ./INFERENCE_CONTRACT.md
COPY models/model_bundle.joblib ./models/model_bundle.joblib

EXPOSE 8000
ENV PYTHONPATH=src

CMD ["uvicorn", "ecom.api.main:app", "--host", "0.0.0.0", "--port", "8000"]