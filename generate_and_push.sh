#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lệnh nào bị lỗi (ví dụ: lỗi LLM API)
set -e

echo "🚀 BẮT ĐẦU SINH DỮ LIỆU (20 docs, 1000 samples)..."
uv run generate_and_train.py --mode generate --num_documents 20 --num_samples 1000

echo "📦 TIẾN HÀNH GIT ADD VÀ COMMIT..."
git add .
git commit -m "Update new generated data"

echo "☁️ ĐANG PUSH LÊN GITHUB (origin master)..."
git push origin master

echo "✅ HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH!"