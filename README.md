## Overview

This application uses a dual-input neural network model to predict how difficult students might find a particular topic. The system analyzes both the text description of the topic and quantifiable features like abstraction level, prerequisites, content volume, and conceptual complexity.

## Features

- **Topic Difficulty Prediction**: Get an estimated difficulty score on a scale of 1-10
- **Teaching Recommendations**: Receive suggestions tailored to the predicted difficulty level
- **Factor Analysis**: See which aspects contribute most to the topic's difficulty
- **Model Training**: Option to retrain the model with synthetic data
- **Interactive UI**: Easy-to-use interface built with Streamlit

## Deep Learning Concepts Used

- **Word Embeddings**: For processing topic descriptions
- **Multi-Input Neural Network**: Combines text and numerical features
- **Dense Layers with Dropout**: For feature extraction and regularization
- **Regression Output**: Linear activation for predicting difficulty score



## Future Improvements

- Collect real teacher-rated data for better model training
- Add more subject-specific features
- Implement user accounts to save previous topic analyses
- Add batch processing for multiple topics

