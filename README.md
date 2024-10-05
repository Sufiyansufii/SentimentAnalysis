# Sentiment Analysis on Twitter Data

# Overview
This project implements a sentiment analysis model that classifies Twitter data into three categories: Positive, Negative, and Neutral. It utilizes deep learning techniques, including Word2Vec for word embeddings and LSTM (Long Short-Term Memory) networks for sequence classification.

# Features
Data preprocessing including tokenization, stop word removal, and stemming.
Word embeddings using Word2Vec.
LSTM-based neural network for sentiment classification.
Evaluation metrics such as accuracy and loss.
User-friendly prediction function for new inputs.
Getting Started
Prerequisites
Python 3.8 or higher
Required libraries:
pandas
numpy
matplotlib
scikit-learn
keras
gensim
nltk
Installation
# Prepare the dataset: 
Place your dataset in the specified directory.
#Run the model: 
Execute the Jupyter notebook to train the model.
# Make predictions: 
Use the provided function to predict sentiment from new text inputs.
# Example of predicting sentiment:

predict("I love the music")
predict("I hate the rain")
# Data Description
The dataset consists of 1,600,000 Twitter posts and contains the following columns:

# target: 
Sentiment label (0: Negative, 2: Neutral, 4: Positive)
# ids: 
Unique tweet identifiers
# date: 
Tweet timestamps
user: Twitter username
text: Tweet content
# Model Architecture
The model consists of the following layers:

# Embedding Layer: 
Uses pre-trained Word2Vec embeddings.
# Dropout Layer: 
Regularization to prevent overfitting.
# LSTM Layer: 
For capturing dependencies in sequential data.
# Dense Layer: 
Output layer for binary classification (sigmoid activation).
# Model Training
The model is trained using the following parameters:

Embedding Size: 300
Epochs: 8
Batch Size: 1024
LSTM Units: 100
After training, the model will be saved as model.h5, and the Word2Vec model and tokenizer will be saved as model.w2v and tokenizer.pkl, respectively.

# Streamlit Application
The Streamlit application allows users to input text and get predictions in real-time. The app uses the trained model to classify input text and display the sentiment label and score.

# Results
The model achieves an accuracy of approximately 79% on the test dataset.
Training and validation accuracy/loss plots are generated to visualize performance.
