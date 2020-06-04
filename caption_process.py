import keras
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.text import Tokenizer
import pickle
import os
from keras.preprocessing.sequence import pad_sequences
with open('filenames.pickle','rb') as f:
    filenames = pickle.load(f)

text_all = []
for filename in filenames:
    path = os.listdir('D:/caption_extraction/'+filename)
    with open('D:/caption_extraction/'+filename+'/'+path[0], 'r', encoding='utf-8') as txt:
        text = txt.read()
        text_all.append(text)

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(text_all)
sequences = tokenizer.texts_to_sequences(text_all)
maxlen = 0
for sequence in sequences:
    if len(sequence) > maxlen:
        maxlen = len(sequence)
data = pad_sequences(sequences, maxlen+16, padding='post')
print(data.dtype)
with open('caption_data.pickle', 'wb') as w:
    pickle.dump(data, w)




