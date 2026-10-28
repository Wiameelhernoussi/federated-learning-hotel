FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /shared/
COPY server/ .
RUN python -m grpc_tools.protoc -I/shared --python_out=. --grpc_python_out=. /shared/fl.proto
CMD ["python", "server.py"]