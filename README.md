# CF-Recommender

This project implements a collaborative filtering-based recommendation system using Pearson correlation to compute item similarity. The system predicts user ratings for items based on historical user-item interaction data.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)

## Introduction

Recommender systems are crucial for many modern applications, providing personalised recommendations to users based on their preferences and behavior. This project demonstrates a basic implementation of a collaborative filtering recommendation system using the Pearson correlation coefficient to compute item similarities. The system is designed to predict ratings that users might give to items they haven't interacted with yet.

## Project Structure

The project contains the following files:

- **`script.py`**: The main Python script 

- **`train_100k_withratings.csv`**: The training dataset, which contains user-item interactions along with their corresponding ratings. This dataset is used to compute item similarities and train the recommendation model.

- **`test_100k_withoutratings.csv`**: The test dataset, which includes user-item interactions without ratings. The model predicts ratings for these interactions, which are then compared with actual ratings to evaluate performance.

- **`submission.csv`**: The output file that contains the predicted ratings for the user-item pairs in the test set.

