import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

class Chatbot:
    def __init__(self, error_threshold=0.35):
        # list of tokenized words
        self.vocabulary = []
        # possible tags for sentences (e.g. greeting)
        self.classes = []
        # lists of patterns for specific tags for training
        self.documents = []
        self.ignore_words = ['?', '!']
        # mapping of responses for patterns by tag (class)
        self.intents = json.loads(open('intents.json').read())
        self.model = Sequential()
        self.ERROR_THRESHOLD = error_threshold
        
    def create_helper_files(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # tokenize sentences
                words = nltk.word_tokenize(pattern)
                voc_words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in self.ignore_words]
                self.vocabulary.extend(voc_words)
                # add documents in the corpus
                self.documents.append((words, intent['tag']))
                # add to our classes set
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # lemmaztize and lower each word and remove duplicates
        self.vocabulary = sorted(list(set(self.vocabulary)))
        # sort classes
        self.classes = sorted(list(set(self.classes)))
       
    def create_training_data(self):
         # create classes and vocabulary
        self.create_helper_files()
        # create our training data
        training = []
        # create an empty array for our output
        output_empty = [0] * len(self.classes)
        # training set, bag of words for each sentence
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            # create our bag of words array with 1, if word match found in current pattern
            for word in self.vocabulary:
                bag.append(1) if word in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
            
        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)
        # create train and test lists. X - patterns, Y - intents
        train_x = list(training[:,0])
        train_y = list(training[:,1])
        print(f"Number of classes: {len(self.classes)}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Training data size: {len(train_x)}")
        print("Training data created")
        return train_x, train_y
        
    def train(self):
        # create training data
        train_x, train_y = self.create_training_data()
        # initialize model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
        # equal to number of intents to predict output intent with softmax
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))

        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        #fitting and saving the model 
        hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        self.model.save('chatbot_model.h5', hist)
        print("model created")
        return self
    
    def clean_up_sentence(self, sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(self.vocabulary)  
        for s in sentence_words:
            for i,w in enumerate(self.vocabulary):
                if w == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    break
        return(np.array(bag))

    def predict(self, sentence):
        # filter out predictions below a threshold
        p = self.bow(sentence)
        res = self.model.predict(np.array([p]))[0]
        print(f"Predictions of class for sentance: {sentence}\n")
        for i,r in enumerate(res):
            print(f"intent: {self.classes[i]}, probabilty: {str(r)}\n")
        results = [[i,r] for i,r in enumerate(res) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        most_likely_result = {"intent": self.classes[results[0][0]], "probabilty": str(results[0][1])}
        return most_likely_result