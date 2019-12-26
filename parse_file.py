from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import codecs
import csv
import os
import pickle


BASE_DIR = 'D:/datasets/SST/'
EM_BASE_DIR = 'D:/glove.42B.300d/'
TRAIN_DATA_FILE = BASE_DIR + 'sst5_train_sentences.txt'#'train_micro.csv'
GLOVE_EMBEDDING = EM_BASE_DIR + 'glove.42B.300d.txt'
TRAIN_RATIO = 0.8
MAX_SEQUENCE_LENGTH = 60
MAX_NB_WORDS = 17000
EMBEDDING_DIM = 300
DATA_DIR = "D:/vae_data/"


texts = []
with codecs.open(TRAIN_DATA_FILE, encoding='ascii') as f:
    reader = csv.reader(f, delimiter='\t')
    for values in reader:
        if len(values[0].split()) <= MAX_SEQUENCE_LENGTH:
            texts.append(values[0])
print('Found %s texts in train.txt' % len(texts))
n_sents = len(texts)


#======================== Tokenize and pad texts lists ===================#
tokenizer = Tokenizer(17000, oov_token='<unk>') #+1 for 'unk' token
tokenizer.fit_on_texts(texts)
print('Found %s unique tokens' % len(tokenizer.word_index))
## **Key Step** to make it work correctly otherwise drops OOV tokens anyway!
tokenizer.word_index[tokenizer.oov_token] = len(tokenizer.word_index)
tokenizer.word_index['<PAD>'] = 0
word2index = tokenizer.word_index.items() #the dict values start from 1 so this is fine with zeropadding
index2word = {v: k for k, v in word2index}
sequences = tokenizer.texts_to_sequences(texts)
data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
print('Shape of data tensor:', data_1.shape)
NB_WORDS = len(word2index) #+1 for zero padding
print('Found %s unique tokens' % len(tokenizer.word_index))




train_data = data_1[:int(len(data_1)*TRAIN_RATIO)]
test_data = data_1[int(len(data_1)*TRAIN_RATIO):]
print("Train data size %d" %len(train_data))
print("Test data size %d" %len(test_data))
print("Train data peak")
print(train_data[:3,:])
vocabs = [index2word[i] for i in range(NB_WORDS)]

with open(DATA_DIR + "vocab", 'wb') as f:
    pickle.dump(vocabs, f)

with open(DATA_DIR + "Train", 'wb') as f:
    pickle.dump(train_data, f)

with open(DATA_DIR + "Test", 'wb') as f:
    pickle.dump(test_data,f)


#======================== prepare GLOVE embeddings =============================#
embeddings_index = {}
f = open(GLOVE_EMBEDDING, encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        continue
f.close()
print('Found %d word vectors.' % len(embeddings_index))

glove_embedding_matrix = np.random.uniform(-1.0,1.0,(NB_WORDS, EMBEDDING_DIM))
unkCnt = 0
for word, i in word2index:
    if i < NB_WORDS: #+1 for 'unk' oov token
        if word == '<PAD>':
            embedding_vector = np.random.uniform(-1.0,1.0,(EMBEDDING_DIM))
        elif word == '<unk>':
            embedding_vector = np.random.uniform(-1.0,1.0,(EMBEDDING_DIM))
        else:
            if word in embeddings_index.keys():
                embedding_vector = embeddings_index[word]
            else:
                unkCnt += 1
        glove_embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(glove_embedding_matrix, axis=1) == 0))
print("unkown tokens %d out of total %d tokens"%(unkCnt, len(word2index)-2))
with open(DATA_DIR + "embedding", 'wb') as f:
    pickle.dump(glove_embedding_matrix, f)


