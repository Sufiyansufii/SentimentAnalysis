from keras.models import load_model
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from gensim.models import Word2Vec 
import pickle

w2v_model = Word2Vec.load('C:/Final Year Project/uttn/uttn/uttn/model.w2v')
model = load_model('C:/Final Year Project/uttn/uttn/uttn/model.h5')
with open('C:/Final Year Project/uttn/uttn/uttn/tokenizer.pkl', 'rb') as handle:
  tokenizer = pickle.load(handle)
with open('C:/Final Year Project/uttn/uttn/uttn/encoder.pkl', 'rb') as handle:
  encoder = pickle.load(handle)

  
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"

SENTIMENT_THRESHOLDS = (0.4, 0.7)
def decode_sentiment(score, include_neutral=True):
  if include_neutral:
    label = NEUTRAL
  if score == SENTIMENT_THRESHOLDS[1]:
    label = POSITIVE
    return label
  else:
    return NEGATIVE if score < 0.5 else POSITIVE

def predict(text, include_neutral=True): 
  start_at = time.time() 
  x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)  
  #Predict
  score = model.predict([x_test])[0]
  #Decode sentiment
  label = decode_sentiment(score, include_neutral=include_neutral)
  return {"label": label, "score": float(score), "elapsed_time": time.time()-start_at}
va=predict("hate")
print(va)