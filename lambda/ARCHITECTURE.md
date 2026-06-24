# Lambda Ingestion Architecture

## Why Lambda for ingestion?

Current (synchronous):
  Browser → POST /ingest → FastAPI waits → ColQwen2.5 embeds all pages → response
  Problem: 200-page filing = 5-10 min browser wait, request timeout risk

Lambda (async):
  Browser → POST /ingest → FastAPI returns "processing" immediately
  In background: S3 PUT → Lambda triggered → ColQwen2.5 embeds → Qdrant write → DynamoDB status update
  Browser polls GET /ingest-status/{doc_id} until status = "complete"
  Result: instant response, background processing, no timeout risk

## Architecture

```
User
 │
 ├── 1. Request presigned S3 URL  →  FastAPI  →  S3 presigned URL
 │
 ├── 2. Upload PDF directly to S3 (browser → S3, no server bandwidth used)
 │
 └── 3. Poll status  →  FastAPI  →  DynamoDB  →  {status: "processing" | "complete"}

S3 PUT event
 │
 └──  Lambda (container image with ColQwen2.5 + CUDA)
       ├── Download PDF from S3
       ├── Render pages (pdf2image)
       ├── Embed with ColQwen2.5 on GPU
       ├── Write multi-vectors to Qdrant
       └── Update DynamoDB: {doc_id, status: "complete", pages: N}
```

## Deployment steps (when ready to go to AWS)

1. Build and push the Lambda container image:
   docker build -t apertura-lambda -f lambda/Dockerfile .
   aws ecr create-repository --repository-name apertura-lambda
   docker tag apertura-lambda <account>.dkr.ecr.<region>.amazonaws.com/apertura-lambda:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/apertura-lambda:latest

2. Create Lambda function from container image in AWS Console:
   - Runtime: Container image
   - Memory: 10240 MB (maximum — ColQwen2.5 needs ~8GB)
   - Timeout: 900 seconds (15 minutes — max Lambda timeout)
   - Ephemeral storage: 10240 MB
   - GPU: Use AWS Lambda Graviton or ml.g4dn via Lambda SnapStart

3. Add S3 trigger: on ObjectCreated in your PDFs bucket

4. Add environment variables:
   QDRANT_URL, QDRANT_API_KEY, COLPALI_MODEL, STATUS_TABLE, AWS_REGION

5. Create DynamoDB table: apertura-ingestion-status (partition key: doc_id)

6. IAM permissions for Lambda:
   - s3:GetObject on your PDFs bucket
   - dynamodb:PutItem on the status table

## Cost estimate (portfolio scale, ~20 ingestions/month)
  Lambda: ~$0.05/month (free tier covers most of it)
  S3:     ~$0.01/month
  DynamoDB: free tier
  Total:  negligible
