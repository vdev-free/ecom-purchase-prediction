FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# install runtime deps (minimal, but enough for your current app)
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
      fastapi uvicorn joblib pandas numpy scikit-learn

COPY src ./src
COPY INFERENCE_CONTRACT.md ./INFERENCE_CONTRACT.md
COPY models/best_model.joblib ./models/best_model.joblib
RUN PYTHONPATH=src python -m ecom.package_model

EXPOSE 8000
ENV PYTHONPATH=src

CMD ["uvicorn", "ecom.api.main:app", "--host", "0.0.0.0", "--port", "8000"]