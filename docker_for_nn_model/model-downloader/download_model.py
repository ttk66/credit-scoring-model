import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def getenv_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"ERROR: missing required env var {name}", file=sys.stderr)
        sys.exit(1)
    return value


def download_with_retries(client, bucket: str, key: str, dest: str, attempts: int = 5) -> None:
    for attempt in range(1, attempts + 1):
        try:
            client.download_file(bucket, key, dest)
            print(f"Downloaded s3://{bucket}/{key} -> {dest}")
            return
        except (ClientError, BotoCoreError) as exc:
            print(f"Download failed (attempt {attempt}/{attempts}): {exc}", file=sys.stderr)
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)


def parse_aux_keys(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def main() -> None:
    access_key = getenv_required("STORAGE_ACCESS_KEY")
    secret_key = getenv_required("STORAGE_SECRET_KEY")
    bucket = getenv_required("MODEL_BUCKET")

    endpoint = os.getenv("STORAGE_ENDPOINT", "https://storage.yandexcloud.net")
    key = os.getenv("MODEL_KEY", "credit_scoring.onnx")
    dest = os.getenv("MODEL_PATH", "/models/credit_scoring.onnx")
    aux_keys = parse_aux_keys(os.getenv("MODEL_AUX_KEYS", ""))

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    download_with_retries(client, bucket, key, dest)

    # Download optional auxiliary files (e.g. ONNX external data/scaler) to model directory.
    model_dir = Path(dest).parent
    for aux_key in aux_keys:
        aux_dest = str(model_dir / Path(aux_key).name)
        download_with_retries(client, bucket, aux_key, aux_dest)


if __name__ == "__main__":
    main()
