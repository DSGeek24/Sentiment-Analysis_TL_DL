# importing required packages
import re
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import itertools
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
from keras.optimizers import Adam
from keras.layers import SpatialDropout1D,Dropout,Conv1D,GlobalMaxPooling1D,MaxPooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
seed=np.random.seed(7)

####################################################
## Loading Data
####################################################

phrase=[]
labels=[]
train=pd.DataFrame.from_csv('IMDB_traindata.tsv', sep='\t', header=0)

# # # load testing data
# with open("reviews.tsv") as testing:
#     test = csv.reader(testing, delimiter="\t", quotechar='"')
#     for s in test:
#         phrase.append(s[0])
#         labels.append(s[1])
#
# del phrase[0]
# del labels[0]
#
# train=pd.DataFrame()
# train['review']=phrase
# train['sentiment']=labels

def clean_text(text):
    words = (re.sub("[^a-zA-Z]", " ", text)).lower()
    words = words.split()
    wordList = re.sub("[^\w]", " ", words).split()
    words = [word for word in wordList if not word in stopwords]
    words = [w for w in wordList if w.lower() not in stop_words]
	words=words.split()
	words= " ".join(words)
	stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in words]
    text = " ".join(stemmed_words)
    return text

train['review'] = train['review'].map(lambda x: clean_text(x))
y = train['sentiment']

#Use this in case of multi labels(not binary polarity)
#target=train.sentiment.values
#y=np_utils.to_categorical(target)

le = LabelEncoder()
encoded_labels = le.fit_transform(train.sentiment.values)
y=encoded_labels

X_train , X_test , Y_train , Y_test = train_test_split(train['review'],y,test_size = 0.20)
print(X_train.head())

# Summarize number of words
print("Number of unique words in the data are {}".format(len(np.unique(np.hstack(X_train)))))

#Find maximum number of words in a review
length = []
for x in X_train:
    length.append(len(x.split(" ")))
print("Max number of words are {}".format(max(length)))

#initialize parameters
max_features = 10000
max_words = 700
batch_size = 32
epochs = 3

#Keras tokenizer and pad sequences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train =pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

#NOTE: In case of binary class, loss=binary_crossentropy, optimizer=adam, metrics=accuracy, Activation function= sigmoid
#In case of multi class, loss=categorical_crossentropy, optimizer=adam, metrics=categorical_accuracy, Activation function= softmax

# #CNN Model#
model_CNN= Sequential()
model_CNN.add(Embedding(max_features,100,input_length=max_words))
model_CNN.add(Dropout(0.2))
model_CNN.add(Conv1D(32,kernel_size=3,padding='same',activation='relu',strides=1))
model_CNN.add(GlobalMaxPooling1D())
model_CNN.add(Dense(128,activation='relu'))
model_CNN.add(Dropout(0.2))
model_CNN.add(Dense(1,activation='sigmoid'))
model_CNN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_CNN.summary()
history=model_CNN.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=epochs, batch_size=batch_size, verbose=1)


#GRU Model#- results were little similar to LSTM so didn't incorporate the results in report

# model_GRU=Sequential()
# model_GRU.add(Embedding(max_features,100,mask_zero=True))
# model_GRU.add(GRU(64,dropout=0.4,return_sequences=True))
# model_GRU.add(GRU(32,dropout=0.5,return_sequences=False))
# model_GRU.add(Dense(num_classes,activation='softmax'))
# model_GRU.compile(loss='binary_crossentropy',optimizer=Adam(lr = 0.001),metrics=[categorical_accuracy])
# model_GRU.summary()
# history=model_GRU.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=epochs, batch_size=batch_size, verbose=1,callbacks=[metrics])


# # # #LSTM Model#
model_LSTM = Sequential()
model_LSTM.add(Embedding(max_features, 100, mask_zero=True))
model_LSTM.add(LSTM(64, dropout=0.4, return_sequences=True))
model_LSTM.add(LSTM(32, dropout=0.5, return_sequences=False))
model_LSTM.add(Dense(1, activation='sigmoid'))
model_LSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_LSTM.summary()
history=model_LSTM.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=epochs, batch_size=batch_size, verbose=1)

# #CNN-LSTM Model#

model_LSTM = Sequential()
model_LSTM.add(Embedding(max_features, 100, input_length=max_words))
model_LSTM.add(Dropout(0.2))
model_LSTM.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model_LSTM.add(MaxPooling1D(pool_size=2))
# # 1 layer of 150 units in the hidden layers of the LSTM cells
model_LSTM.add(LSTM(150))
model_LSTM.add(Dense(1, activation='sigmoid'))
model_LSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_LSTM.summary())

history = model_LSTM.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size,verbose=1)

##CNN LSTM using GloVe word embeddings 
embeddings_index = dict()
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
     values = line.split()
     word = values[0]
     coefs = np.asarray(values[1:], dtype='float32')
     embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((max_features, 100))
for word, index in tokenizer.word_index.items():
    if index > max_features - 1:
         break
	else:
         embedding_vector = embeddings_index.get(word)
         if embedding_vector is not None:
             embedding_matrix[index] = embedding_vector

# ## create model
model_glove = Sequential()
model_glove.add(Embedding(max_features, 100, input_length=max_words, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# ## Fit train data
history = model_glove.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=batch_size,verbose=1)

#Plots and Performance metrics#

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


## multiclass or binary report
## If not binary, binary parameter is set to False to get multi class report
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=True):
    if not binary:
        y_true = np.argmax(y_true, axis=1)

    y_pred = model.predict_classes(x, batch_size=batch_size)

    print("Accuracy : " + str(accuracy_score(y_true, y_pred)))

    print("")

    print("Classification Report")
    print(classification_report(y_true, y_pred, digits=5))

    cnf_matrix = confusion_matrix(y_true, y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix, classes=classes)

##Plotting- Training and Validation loss####
#
fig1 = plt.figure()
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss curves',fontsize=16)
fig1.savefig('loss_cnn.png')
plt.show()
#
# ##Plotting-Training and Validation accuracy##
fig2=plt.figure()
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
fig2.savefig('accuracy_cnn.png')
plt.show()

#get the classification report and evaluate and report accuracy
le = LabelEncoder()
encoded_labels = le.fit_transform(train.sentiment.values)
full_multiclass_report(model3_LSTM,
                        X_test,
                        Y_test,
                       le.inverse_transform(np.arange(2)))

scores = model_LSTM.evaluate(X_test,Y_test,verbose=0)
print(scores)
print("Accuracy: %.2f%%" % (scores[1]*100))