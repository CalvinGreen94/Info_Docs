# Trading Prediction System

## Overview

This repository contains scripts for a trading prediction system that utilizes machine learning models to predict price changes, RSI, and other trading metrics. The system performs real-time analysis, decision-making, and data saving for trading actions.

## Components

The project consists of three main parts:

1. **Data Preparation**:
   - **Purpose**: Prepares and preprocesses data for training and testing.
   - **Features**:
     - Data cleaning and feature extraction.
     - Normalization and scaling of features.
     - Splitting data into training and testing sets.

2. **Model Training and Prediction Script**:
   - **Purpose**: Trains machine learning models and makes predictions based on the prepared data.
   - **Features**:
     - Fits multiple regression models to training data.
     - Makes predictions on test data for price changes, RSI, and other metrics.
     - Evaluates model performance using score metrics.

3. **Trading Execution Script**:
   - **Purpose**: Executes trading decisions based on model predictions and real-time data.
   - **Features**:
     - Decision-making logic for buying or selling based on predictions and market conditions.
     - Data saving to JSON files and potentially a database.
     - Error handling and logging for robust execution.
