FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy scipy h5py pandas onnxruntime fastapi "uvicorn[standard]" pydantic
COPY src/ src/
COPY artifacts/ artifacts/
COPY data/manifest.csv data/manifest.csv
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "driftguard.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
