#!/bin/bash

# HuggingFace LIBERO 데이터셋 다운로드 스크립트

DATASET_ID="HuggingFaceVLA/libero"
LOCAL_DIR="/workspace/data/libero"

echo "==================================="
echo "LIBERO Dataset Download"
echo "==================================="
echo "Dataset: $DATASET_ID"
echo "Target: $LOCAL_DIR"
echo ""

# HuggingFace CLI 체크
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found"
    echo "Install with: pip install huggingface-hub"
    exit 1
fi

# 데이터 디렉토리 생성
mkdir -p $LOCAL_DIR

# 다운로드 실행
echo "Starting download..."
huggingface-cli download \
    $DATASET_ID \
    --repo-type dataset \
    --local-dir $LOCAL_DIR

echo ""
echo "✓ Download complete!"
echo "Dataset location: $LOCAL_DIR"

