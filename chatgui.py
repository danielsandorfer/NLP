import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import tkinter
from tkinter import *

class ChatGui:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.intents = json.loads(open('intents.json').read())

    def get_response(self, intent):
        tag = intent['intent']
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if(i['tag'] == tag):
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, msg):
        intent = self.chatbot.predict(msg)
        res = self.get_response(intent)
        return res