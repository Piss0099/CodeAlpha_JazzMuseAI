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
  
