# JazzMuseAI: AI Music Generator using LSTM
# Author: Ravi Choudhary
# Internship: CodeAlpha AI Internship (2025)

import os
import numpy as np
from music21 import converter, stream
from keras.models import Sequential
from keras.layers import LSTM, Dense
import random

# Collect notes from MIDI files
notes = []
for file in os.listdir('data'):
    if file.endswith(".mid"):
        midi = converter.parse(os.path.join('data', file))
        for el in midi.flat.notes:
            notes.append(str(el.pitch))

if not notes:
    raise ValueError("⚠️ No MIDI files found in 'data/' folder. Please add some jazz/classical MIDI files.")
  
# Mapping notes
pitches = sorted(set(notes))
note_to_int = {note: n for n, note in enumerate(pitches)}
encoded_notes = [note_to_int[n] for n in notes]

# Dataset prep
seq_length = 20
X, y = [], []
for i in range(len(encoded_notes) - seq_length):
    X.append(encoded_notes[i:i + seq_length])
    y.append(encoded_notes[i + seq_length])

X = np.array(X).reshape(len(X), seq_length, 1)
y = np.array(y)

# Model
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, 1)))
model.add(Dense(len(pitches), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train
model.fit(X, y, epochs=20, batch_size=64)

# Generate new melody
start_index = random.randint(0, len(X) - 1)
generated = list(X[start_index].flatten())

for _ in range(50):
    prediction = model.predict(np.array(generated[-seq_length:]).reshape(1, seq_length, 1))
    next_note = np.argmax(prediction)
    generated.append(next_note)

# Convert back to notes
generated_notes = [pitches[i] for i in generated]

# Save to MIDI
output_stream = stream.Stream()
for pitch in generated_notes:
    output_stream.append(converter.note.Note(pitch))
output_stream.write('midi', fp='generated_music.mid')

print("✅ Music generated and saved as 'generated_music.mid'")
