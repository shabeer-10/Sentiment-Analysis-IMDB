# Sentiment Analysis on IMDB Dataset

This project implements a sentiment analysis model using the IMDB dataset. The model is built with TensorFlow and Keras, leveraging an LSTM network for natural language processing tasks.

## Features

- **Dataset**: IMDB movie reviews dataset with 10,000 most frequent words.
- **Model Architecture**:
  - Embedding Layer
  - LSTM Layer
  - Dense Layers with Dropout
  - Sigmoid Output for Binary Classification
- **Preprocessing**: Tokenizes and pads sequences for uniform input length.
- **Functionality**:
  - Trainable for binary sentiment classification.
  - Real-time predictions based on user input.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy

Install dependencies with:
```bash
pip install tensorflow numpy
```

## Usage

1.Clone the repository:
```bash
git clone https://github.com/shabeer-10/Sentiment-Analysis-IMDB.git
cd Sentiment-Analysis-IMDB
```
Run the script:
```bash
python SentimentAnalysis.py
```

## Model Training

• The model is trained on the IMDB dataset, with 20% of training data used for validation.
• Training details:
   • Epochs: 5
   • Batch size: 512

## Example

```text
Enter a review to analyze sentiment: This movie was absolutely fantastic!
Sentiment: positive (Confidence: 0.95)
```

## Results

• Test Accuracy: ~85% (on IMDB test set)

## Project Structure

```
Sentiment-Analysis-IMDB/
│
├── SentimentAnalysis.py   # Main script for training, evaluation, and predictions
├── README.md              # Project documentation
```















