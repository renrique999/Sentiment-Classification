# ==========================================
# Sentiment Analysis using LSTM
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense
)

# ==========================================
# Load Dataset
# ==========================================

columns = [
    'target',
    'id',
    'date',
    'flag',
    'user',
    'text'
]

data = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    names=columns
)

print("\nDataset Loaded Successfully!")

# ==========================================
# Use Smaller Dataset for Faster Training
# ==========================================

data = data[['target', 'text']]

data = data.head(10000)

# Convert labels
data['target'] = data['target'].replace(4, 1)

# ==========================================
# Tokenization
# ==========================================

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(data['text'])

sequences = tokenizer.texts_to_sequences(data['text'])

print("\nTokenization Completed!")

# ==========================================
# Padding
# ==========================================

X = pad_sequences(sequences, maxlen=100)

y = data['target']

print("\nPadding Completed!")

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Build LSTM Model
# ==========================================

model = Sequential()

model.add(
    Embedding(
        input_dim=5000,
        output_dim=128,
        input_length=100
    )
)

model.add(LSTM(64))

model.add(Dense(1, activation='sigmoid'))

# ==========================================
# Compile Model
# ==========================================

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nModel Compiled Successfully!")

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# ==========================================
# Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nAccuracy:", accuracy)

# ==========================================
# Plot Accuracy Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['Train', 'Validation'])

plt.show()

# ==========================================
# Plot Loss Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(['Train', 'Validation'])

plt.show()