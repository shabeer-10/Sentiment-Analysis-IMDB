import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Pad sequences to ensure uniform input length (100 words)
max_length = 100
train_data = pad_sequences(train_data, maxlen=max_length)
test_data = pad_sequences(test_data, maxlen=max_length)

# Build the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_length), # Embedding layer
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),                       # LSTM layer
    Dense(32, activation='relu'),                                       # Dense layer
    Dropout(0.5),                                                       # Dropout for regularization
    Dense(1, activation='sigmoid')                                      # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels,
                    epochs=5,
                    batch_size=512,
                    validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_accuracy:.4f}")

# Function to preprocess input text and make predictions
def preprocess_text(input_text):
    # Load the IMDB word index dictionary
    word_index = imdb.get_word_index()
    
    # Convert the input text to lowercase and tokenize
    input_tokens = [word_index.get(word, 0) for word in input_text.lower().split()]
    
    # Pad the tokenized input to the same length as the training data
    padded_input = pad_sequences([input_tokens], maxlen=max_length)
    
    return padded_input

# Predict sentiment based on user input
def predict_sentiment(input_text):
    # Preprocess the input text
    processed_input = preprocess_text(input_text)
    
    # Predict sentiment
    prediction = model.predict(processed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")

# Example usage with user input
user_input = input("Enter a review to analyze sentiment: ")
predict_sentiment(user_input)

