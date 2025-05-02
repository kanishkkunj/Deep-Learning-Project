# Topic Difficulty Recommender for Teachers

A deep learning-based web application that helps teachers estimate the difficulty level of academic topics based on various factors.

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

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/topic-difficulty-recommender.git
   cd topic-difficulty-recommender
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Fill in the details about your topic and click "Predict Difficulty"

## Project Structure

- `app.py`: Main Streamlit application
- `model.py`: Neural network architecture and training functions
- `data_utils.py`: Data generation and preprocessing utilities
- `requirements.txt`: Project dependencies

## Future Improvements

- Collect real teacher-rated data for better model training
- Add more subject-specific features
- Implement user accounts to save previous topic analyses
- Add batch processing for multiple topics

## License

MIT

## Acknowledgements

This project was created as part of a college deep learning course.