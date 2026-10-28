FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu
COPY client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /shared/
COPY client/ .
# hotel_dashboard.html is included in client/ above — served by hotel_api.py on port 8000
RUN python -m grpc_tools.protoc -I/shared --python_out=. --grpc_python_out=. /shared/fl.proto
EXPOSE 8000
CMD ["python", "client.py"]