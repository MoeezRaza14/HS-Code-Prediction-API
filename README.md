# Product Classification API

This repository contains a FastAPI application that provides an API for predicting HS codes for given product descriptions. 
The API utilizes a neural network model, a TF-IDF vectorizer, and a label encoder for making predictions.

## Overview

- **Model**: A neural network model saved in `model.h5`.
- **Vectorizer**: TF-IDF vectorizer saved in `vectorizer.pkl`.
- **Label Encoder**: Label encoder saved in `label_encoder.pkl`.

## API Endpoints

### POST /predict

This endpoint accepts a list of product descriptions and returns the top 3 HS codes with their respective probabilities.

#### Request

**URL**: `/predict`  
**Method**: `POST`  
**Content-Type**: `application/json`

**Request Body**:

```json
{
    "descriptions": [
        "Product description 1",
        "Product description 2"
    ]
}
