# from repo root
docker build -t us-docker.pkg.dev/$PROJECT_ID/mia/mia:latest .

# or Cloud Build
gcloud builds submit --tag us-docker.pkg.dev/$PROJECT_ID/mia/mia:latest .


# Then deploy to Cloud Run GPU
gcloud run deploy mia \
  --image us-docker.pkg.dev/$PROJECT_ID/mia/mia:latest \
  --region us-central1 \
  --gpu-type nvidia-l4 --gpu-count 1 \
  --cpu 4 --memory 16Gi \
  --concurrency 1 \
  --min-instances 1 \
  --set-env-vars OUTPUT_BUCKET=mia-results
