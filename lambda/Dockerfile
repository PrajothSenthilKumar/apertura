# Lambda container image for ColQwen2.5 ingestion
# Build: docker build -t apertura-lambda -f lambda/Dockerfile .
# Push:  docker tag apertura-lambda <account>.dkr.ecr.<region>.amazonaws.com/apertura-lambda:latest
#        docker push <account>.dkr.ecr.<region>.amazonaws.com/apertura-lambda:latest

FROM public.ecr.aws/lambda/python:3.12

RUN dnf install -y poppler-utils && dnf clean all

RUN pip install torch==2.6.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu126 \
    --no-cache-dir

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir \
    "colpali-engine==0.3.9" \
    "transformers>=4.50.0,<4.51.0" \
    boto3

COPY lambda/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
