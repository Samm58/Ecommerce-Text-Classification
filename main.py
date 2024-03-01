#%%
# -- 1. SETUP --
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn, os, io, json, pickle
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
#%%
# -- 2. DATA LAODING --

PATH = os.path.join(os.getcwd(), 'ecommerceDataset.csv')
df = pd.read_csv(PATH)

#%%
# -- 3. DATA EXPLORATORY --

# The dataset do not have designed columns, so we assign manually
df.columns = ['category', 'product_description']

df.info()
print("Shape of the data:", df.shape)
print('--------------------------------------------------------')
print("Data description:\n" , df.describe().transpose())
print('--------------------------------------------------------')
print("Example data:\n", df.head(1))
print('--------------------------------------------------------')
print("NA values:\n", df.isna().sum())
print('--------------------------------------------------------')
print("Duplicates:\n", df.duplicated().sum())
print('--------------------------------------------------------')
print("Values counts:\n", df['category'].value_counts())

# -- FINDINGS --
# 1. There's one NaN values in the product_description columns, might as well drop it
# 2. Some of the product_description contains float, need to change into string

# %%
# -- 4. DATA PREPROCESSING --

# 1.    Drop the NaN row
df.dropna(inplace=True)

# 2.    Isolate the feature and labels
features = df['product_description'].values
labels = df['category'].values

# 3.    Perform label encoding on category column
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

#%%
# -- 5. TRAIN TEST SPLIT --

X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, train_size=0.8, random_state=42)

#%%
# -- 6. TOKENIZATION --
# Define hyperparameters
vocab_size = 5000
oov_token = "<OOV>"
max_length = 200
embedding_dim = 64

# Define the Tokenizer object
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=vocab_size, 
    oov_token=oov_token,
    split=' '
)

# Convert elements into string
X_train = X_train.astype(str)
tokenizer.fit_on_texts(X_train)

# Inspection on the Tokenizer
word_index = tokenizer.word_index
word_index

# Use the tokenizer to tranform text to tokens
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
print(X_train[0])
print(X_test_tokens[0])

#%%
# -- 7. PADDING AND TRUNCATING --
X_train_padded = keras.utils.pad_sequences(
    X_train_tokens,
    maxlen=max_length,
    padding='post',
    truncating='post'
)

X_test_padded = keras.utils.pad_sequences(
    X_test_tokens,
    maxlen=max_length,
    padding='post',
    truncating='post'
)

print(X_train_padded.shape)
# # 200 is after we set the max length

# Create a function that can decode the tokens
# 1.    Create a reversed word index
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

reversed_word_index

# Create the function for the decoding
def decode_tokens(tokens):
    return " ".join([reversed_word_index.get(i,"?") for i in tokens])

print(X_train[3])
print('-------------------------------------')
print(decode_tokens(X_train_padded[3]))

#%%
# -- 8. MODEL DEVELOPMENT --

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, embedding_dim))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False, kernel_regularizer=keras.regularizers.L2(l2=0.01))))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(len(np.unique(labels)), activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(patience=5)
logpath = os.path.join('tensorboard_log', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = keras.callbacks.TensorBoard(logpath)
max_epoch = 10

#%%
# -- 9. MODEL TRAINING --
history = model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=max_epoch, callbacks=[early_stopping, tb])

# %%
# -- 10. RESULTS --

# Plot the graphs of loss and accuracy
# 1.    Loss graph
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss, Validation Loss'])
plt.show()

# 2.    Accuracy graph
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()

# Accuracy and F1 Score
y_pred_classes = np.argmax(model.predict(X_test_padded), axis=1)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred_classes)

print('Accuracy: ', accuracy)

y_pred = model.predict(X_test_padded)
y_pred_classes = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print('F1 Score: ', f1)

#%%
# -- 11. MODEL SAVING --

tokenizer_json = tokenizer.to_json()
with io.open('saved_model/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

model.save(os.path.join('saved_model', 'classify.h5'))

# %%
# Model Architecture
# tf.keras.utils.plot_model(model)
