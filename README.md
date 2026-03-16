# Collaborative Filtering Recommender System

## Project Summary

Built a user-based collaborative filtering recommendation system in Python as a final-year Computer Science project at the University of Southampton.
The system predicts ratings for unseen user-item pairs using Pearson correlation to identify similar users.

## How It Works

1. **Load** the training data — 100,000 user-item ratings
2. **Compute similarity** between users using the Pearson correlation coefficient
3. **Predict ratings** for unseen pairs by taking a weighted average of the most similar users' ratings
4. **Output** predictions to `submission.csv`

## Why Pearson Correlation

Pearson accounts for differences in individual rating scales — a user who always rates 4–5 and one who rates 1–3 can still be identified as similar, where simpler metrics like cosine similarity would miss this.

## Tools

Python 3 | pandas | numpy | GitHub

## Files

| File | Description |
|---|---|
| `script.py` | Main implementation |
| `train_100k_withratings.csv` | Training data: 100k user-item ratings |
| `test_100k_withoutratings.csv` | Test data: pairs to predict |
| `submission.csv` | Predicted ratings output |
