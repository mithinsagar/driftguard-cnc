FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==2.* scipy h5py pandas onnxruntime fastapi "uvicorn[standard]" pydantic
COPY src/ src/
COPY artifacts/ae_int8.onnx artifacts/calibration.json artifacts/demo_streams.json artifacts/
ENV PYTHONPATH=/app/src
EXPOSE 8000
# Replay of single recordings needs the dataset: mount it with docker compose (see docker-compose.yml).
CMD ["uvicorn", "driftguard.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
