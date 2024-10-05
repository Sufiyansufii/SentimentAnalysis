# Sentiment Analysis on Twitter Data

# Overview
This project implements a sentiment analysis model that classifies Twitter data into three categories: Positive, Negative, and Neutral. It utilizes deep learning techniques, including Word2Vec for word embeddings and LSTM (Long Short-Term Memory) networks for sequence classification.

## Features
- **Text Preprocessing**: Cleaning and preparing text data for modeling.
- **Word2Vec Integration**: Utilizing Word2Vec embeddings for word representation.
- **LSTM Model**: Building and training a neural network for sentiment classification.
- **Streamlit Interface**: A user-friendly interface for inputting text and viewing predictions.

## Technologies Used
- **Python**
- **Pandas**: For data manipulation and analysis.
- **Keras**: For building and training the neural network model.
- **Gensim**: For Word2Vec model training.
- **NLTK**: For natural language processing tasks.
- **Streamlit**: For creating a web application.
- **Matplotlib**: For data visualization.
  
# Prepare the dataset: 
Place your dataset in the specified directory.
# Run the model: 
Execute the Jupyter notebook to train the model.
# Make predictions: 
Use the provided function to predict sentiment from new text inputs.
# Example of predicting sentiment:

predict("I love the music")
predict("I hate the rain")
# Data Description
The dataset consists of 1,600,000 Twitter posts and contains the following columns:
`target`: Sentiment label (0 for Negative, 2 for Neutral, 4 for Positive)
`ids`: Tweet IDs
`date`: Date of the tweet
`flag`: Unused field
`user`: User who posted the tweet
`text`: The tweet text

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
