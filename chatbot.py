import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, Input, concatenate
from tensorflow.keras.optimizers import SGD, Adam
import random
from nltk.corpus import stopwords
import string
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('punkt')
nltk.download('stopwords')

class Chatbot:
    def __init__(self, error_threshold=0.3, tokenizer_maxlen=35, embedding_size=25):
        # possible tags for sentences (e.g. greeting)
        self.classes = []
        # mapping of responses for patterns by tag (class)
        self.intents = json.loads(open('intents.json').read())
        self.model = Sequential()
        self.ERROR_THRESHOLD = error_threshold
        self.lemmatizer = WordNetLemmatizer()
        self.exclude_words = []
        self.tokenizer = Tokenizer()
        self.vocabulary_size = 0
        self.tokenizer_maxlen = tokenizer_maxlen
        self.embedding_size = embedding_size
        
    def initialize_model(self):
        # initialize model - first is the embedding to 24 dimension, then 2 layers of bidirectional LSTMs for context memorization,
        # then dense layers with dropouts to stop overfitting
        self.model.add(Embedding(self.vocabulary_size, self.embedding_size, input_length=self.tokenizer_maxlen))
        self.model.add(Bidirectional(LSTM(36, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(24)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.classes), activation='softmax'))
        # Compile model
        adam = Adam(learning_rate=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        
    def decontract(self, phrase):
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase) 
        return phrase
        
    def clean_pattern(self, pattern):
        # decontract contracted forms
        decontractated_pattern = self.decontract(pattern)
        # remove non-alphanumeric signs
        non_alpha_pattern = re.sub('[^A-Za-z0-9]+', ' ', decontractated_pattern)
        # tokenize sentences
        words = nltk.word_tokenize(non_alpha_pattern)
        voc_words = [self.lemmatizer.lemmatize(w.lower()) for w in words if w not in self.exclude_words]
        return ' '.join(voc_words)
        
    def create_helper_files(self):
        # lists of training data and labels
        x_train = []
        y_train = []
        # exclude stop words ("the", "a", etc.) and punctuation
        stop_words = set(stopwords.words('english'))
        for character in set(string.punctuation):
            self.exclude_words.append(character)
        self.exclude_words.extend(list(stop_words))
        # intents is a list of objects containing patterns (questions) and responses (possible answers)
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                cleaned_pattern = self.clean_pattern(pattern)
                # add documents in the corpus to the training set
                x_train.append(cleaned_pattern)
                y_train.append(intent['tag'])
                # add names of classes (e.g., greeting)
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        # sort classes
        self.classes = sorted(list(set(self.classes)))
        return x_train, y_train
       
    def format_training_data(self):
         # create classes and vocabulary
        x_train, y_train = self.create_helper_files()
        # create our training data, text data is tokenized
        self.tokenizer.fit_on_texts(x_train)
        training_sequences = self.tokenizer.texts_to_sequences(x_train)
        # all rows padded to size of 50
        training_data = pad_sequences(training_sequences, maxlen=self.tokenizer_maxlen, padding='post', truncating='post')
        self.vocabulary_size = len(self.tokenizer.word_index) + 1
        # create labels for training data
        training_labels = []
        # create an empty array for our output
        output_empty = [0] * len(self.classes)
        # form one-hot encoding of the label
        for tag in y_train:
            output_row = list(output_empty)
            output_row[self.classes.index(tag)] = 1
            # training data contains the one-hot encoding for the class (tag)
            training_labels.append(output_row)           
        # shuffle the texts and labels and turn into numpy arrays
        combined = list(zip(training_data, training_labels))
        random.shuffle(combined)
        training_data, training_labels = zip(*combined)
        training_data = np.array(training_data)
        training_labels = np.array(training_labels)
        print(f"Number of classes: {len(self.classes)}")
        print(f"Vocabulary size: {self.vocabulary_size}")
        print(f"Training data size: {len(training_data)}")
        print("Training data created")
        self.initialize_model()
        print("Model initialized")
        return training_data, training_labels
        
    def train(self, epochs=30, batch_size=5, verbose=1):
        x_train, y_train = self.format_training_data() 
        #fitting and saving the model 
        hist = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.model.save('chatbot_model_3.h5', hist)
        print("model created")
        return self

    def create_sequence(self, sentence):
        cleaned_sentence = self.clean_pattern(sentence)
        sequence = self.tokenizer.texts_to_sequences([cleaned_sentence])
        # all rows padded to size of 50
        padded_sequence = pad_sequences(sequence, maxlen=self.tokenizer_maxlen, padding='post', truncating='post')
        return np.array(padded_sequence)

    def predict(self, sentence):
        # filter out predictions below a threshold
        sequence = self.create_sequence(sentence)
        res = self.model.predict(sequence)[0]
        print(f"\nPredictions of class for sentance: {sentence}\n")
        for i,r in enumerate(res):
            print(f"intent: {self.classes[i]}, probabilty: {str(r)}\n")
        results = [[i,r] for i,r in enumerate(res) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        most_likely_result = {"intent": self.classes[results[0][0]], "probabilty": str(results[0][1])}
        return most_likely_result
