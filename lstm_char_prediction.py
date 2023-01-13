#!/usr/bin/env python3

### This script is meant to be used as CHARACTER predictions in text generation
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import numpy as np
import sys
import data_handling as dh



def create_dataset(ntweets, user):
    
    tweets = dh.web_scrape_tweets(ntweets, user)
    clean_tweets = dh.pre_process(tweets)
    
    
    corpus = " ".join(clean_tweets).lower()
    
    return corpus


def mappings(txt):
    chrs = sorted(list(set(txt)))
    
    chrs_map = dict((char, ind) for ind, char in enumerate(chrs))
    inv_map = dict((ind, char) for ind, char in enumerate(chrs))
    
    return len(chrs), chrs_map, inv_map 


def create_sents(sq_length, txt):
    sents = []
    next_char = []
    
    for i in range(0, len(txt) - sq_length):
            sents.append(txt[i:i + sq_length])
            next_char.append(txt[i + sq_length])

    return sents, next_char

def vectorize_sents(sequences, sq_length, vocab_size, chr_map, target):
    
    X = np.zeros((len(sequences), sq_length, vocab_size))
    y = np.zeros((len(sequences), vocab_size))
    
    for ind, sentence in enumerate(sequences):
        for pos, char in enumerate(sentence):
            X[ind, pos, chr_map[char]] = 1
        y[ind, chr_map[target[ind]]] = 1
    
    return X, y
    
    

def build_model(sq_length, vocab_size, num_lstm_layers, lstm_units = 180, drop_rate = 0.1, 
                optim = 'rms', l_r = 0.01, bidirection = False):
    
    
    
    model = Sequential()
    
    # Optimizer 
    if optim == 'sgd':
        opt = SGD(learning_rate = l_r)
    elif optim == 'adam':
        opt = Adam(learning_rate = l_r)
    elif optim == 'rms':
        opt = RMSprop(learning_rate = l_r)
    
    # LSTM layer(s)
    
    if bidirection == True:
    
        if num_lstm_layers > 1:
        
            model.add(Bidirectional(LSTM(units = lstm_units, return_sequences = True), input_shape = (sq_length, vocab_size)))
            model.add(Dropout(rate = drop_rate))
            
            for i in range(num_lstm_layers - 2):
                model.add(Bidirectional(LSTM(units = lstm_units, return_sequences = True)))
                model.add(Dropout(rate = drop_rate))
            
            model.add(Bidirectional(LSTM(units = lstm_units, return_sequences = False)))
        else:
            model.add(Bidirectional(LSTM(units = lstm_units, return_sequences = False), input_shape = (sq_length, vocab_size)))
     
    else:
        if num_lstm_layers > 1:
        
            model.add(LSTM(units = lstm_units, return_sequences = True, input_shape = (sq_length, vocab_size)))
            #model.add(Dropout(rate = drop_rate))
            
            for i in range(num_lstm_layers - 2):
                model.add(LSTM(units = lstm_units, return_sequences = True))
                #model.add(Dropout(rate = drop_rate))
            
            model.add(LSTM(units = lstm_units, return_sequences = False))
        else:
            model.add(LSTM(units = lstm_units, return_sequences = False, input_shape = (sq_length, vocab_size)))
        
    
    # Add Output Layer
    model.add(Dense(units = vocab_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, 
                  metrics=['accuracy']
                  )
    
    return model


def plot_results(history):
    loss, acc = history.history.values()

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    
    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    
    plt.show()



def tweet_generator(model,txt, sq_length, tweet_length, vocab_size, chr_map, i_map):
    
    tweet = ''
    start = np.random.randint(0, len(txt) - sq_length)
    initial_seed = txt[start: start + sq_length]
    tweet += initial_seed
    
    print("Seed for the tweet:", tweet, "\n")
    
    
    for i in range(tweet_length):
        Xp = np.zeros((1, sq_length, vocab_size))
        for pos, char in enumerate(initial_seed):
            Xp[0, pos, chr_map[char]] = 1
            
        
        pred_chr = model.predict(Xp).flatten()
        ind = np.random.choice(len(pred_chr), p=pred_chr)
        #ind = np.argmax(pred_chr)
        tweet += i_map[ind]
        initial_seed = initial_seed[1:] + i_map[ind] 
        sys.stdout.write(i_map[ind])
       
    return tweet

text = create_dataset(50, 'strandhall')
size_vocab, c_map, inv_map = mappings(text)
Xs, ys = create_sents(sq_length = 60, txt = text)
X, y = vectorize_sents(sequences = Xs, sq_length = 60, vocab_size = size_vocab, chr_map = c_map, target = ys)

callback = EarlyStopping(monitor='loss', min_delta = 0.01, patience=3)
model = build_model(sq_length = 60, vocab_size = size_vocab, bidirection=False, 
                    num_lstm_layers = 1, optim ='rms')

history = model.fit(X, y, batch_size = 120, epochs = 100, callbacks=[callback])

tweet_generator(model = model, txt = text, sq_length = 60, tweet_length = 280, vocab_size = size_vocab, 
                chr_map = c_map, i_map = inv_map)



















