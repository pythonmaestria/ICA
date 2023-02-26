from scipy.io import wavfile
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile as wf



f = sf.SoundFile('talk.wav')
print('Talk samples = {}'.format(f.frames))
print('sample rate = {}'.format(f.samplerate))
print('seconds = {}'.format(f.frames / f.samplerate))



f = sf.SoundFile('success.wav')
print('Success samples = {}'.format(f.frames))
print('sample rate = {}'.format(f.samplerate))
print('seconds = {}'.format(f.frames / f.samplerate))