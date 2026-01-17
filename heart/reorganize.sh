#!/bin/bash

set -e

echo "🚀 Starting ml-models repository reorganization..."

# Create main folders
mkdir -p projects notebooks utils

# ---------------- Linear Regression ----------------
echo "📈 Organizing Linear Regression project..."
mkdir -p projects/linear-regression/data projects/linear-regression/src

mv linear_regression.py projects/linear-regression/src/ 2>/dev/null || true
mv salary_data.csv projects/linear-regression/data/ 2>/dev/null || true
mv capping_flooring_practice_multi.csv projects/linear-regression/data/ 2>/dev/null || true

# ---------------- Decision Tree ----------------
echo "🌳 Organizing Decision Tree project..."
mkdir -p projects/decision-tree/data projects/decision-tree/src

mv decision_tree.py projects/decision-tree/src/ 2>/dev/null || true
mv random_forest.py projects/decision-tree/src/ 2>/dev/null || true
mv titanic.csv projects/decision-tree/data/ 2>/dev/null || true

# ---------------- SVD Image Compression ----------------
echo "🖼️ Organizing SVD Image Compression project..."
mkdir -p projects/image-compression-svd/images projects/image-compression-svd/src

mv svd.py projects/image-compression-svd/src/svd_compression.py 2>/dev/null || true

# ---------------- Heart Disease Project ----------------
echo "❤️ Organizing Heart Disease Prediction project..."
mkdir -p projects/heart-disease-prediction/{data,models,src,api,app}

# ---------------- Cleanup ----------------
echo "🧹 Cleaning cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "✅ Reorganization completed successfully!"
echo "➡️  Run: git status"
echo "➡️  Then: git commit -m \"Reorganize repository professionally\""