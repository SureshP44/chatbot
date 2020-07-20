# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:06:58 2020

@author: Suresh
"""


from flask import Flask, render_template, request
import random 
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem
import re
import pymongo 
from pymongo import MongoClient




cluster = MongoClient("mongodb+srv://suresh:suresh@cluster0.jxrmw.mongodb.net/<dbname>?retryWrites=true&w=majority")

db = cluster['test']
collection  = db['test']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

f = open('intent.json', 'r')    
intents = json.load(f)
    
FILE = "data.pth"
data = torch.load(FILE)

input_size = data['input_size']  
hidden_size = data['hidden_size']
output_size = data['output_size']   
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])

def chat():
    
    user_responses = []
    bot_responses = []
    sentence1 = request.form['user_input']
    user_responses.append(sentence1)
    sentence = tokenize(sentence1)
    stemmed_words = [ stem(w) for w in sentence]
    no_of_pizza=0
    order_id = 0
    sts = ['Your food is being prepared', 'Our executive is out for delivery', 'Sorry for the inconvinence.. you will get you order within 10 mins']
    order_sts = sts[random.randint(0,2)]
    for w in stemmed_words:
        if w == 'order' or w =='want' or w == 'need':
            for wrd in stemmed_words:
                if re.match('^[0-9]*$', wrd):
                    no_of_pizza = int(wrd)
                    choices = list(range(100))
                    random.shuffle(choices)
                    order_id = choices.pop()
                    sts = ['Your food is being prepared', 'Our executive is out for delivery', 'Sorry for the inconvinence.. you will get you order within 10 mins']
                    order_sts = sts[random.randint(0,2)]
                    order_details = {'_id': order_id , 'Address':'none', 'Status': order_sts}
                    collection.insert_one(order_details)
                    bot = f"your order has been recorded and your order id is {order_id}, kindly provide us delivery details"
                    return render_template('index.html', user_input=sentence1, bot_response = bot )
        elif w == 'address':
            result = collection.update({"_id":order_id}, {"$set":{"Address": sentence1}})
            bot = f"your delivery details are recorded status of your order: {order_sts}"
            return render_template('index.html', user_input=sentence1, bot_response = bot )
        elif w == 'statu':
            results = collection.find_one({"_id":order_id})
            if results == 'None':
                bot = 'No orderFound in this id'
                return render_template('index.html', user_input=sentence1, bot_response = bot )
            else:
                bot = results.get('Status')
                return render_template('index.html', user_input=sentence1, bot_response = bot )
                
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim =1)
    prob = probs[0][predicted.item()] 
    
       
    for intent in intents['intents']:
            if tag == intent['tag']:
                bot = random.choice(intent["responses"])
                return render_template('index.html', user_input=sentence1, bot_response = bot )

@app.errorhandler(500)
def errors(e):
    return render_template('error.html', error = 'Oops..! Some error occured.')

if __name__ == '__main__':
    
    app.run(debug = True)
