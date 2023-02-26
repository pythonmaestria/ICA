from scipy.io import wavfile
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile as wf


data, samplerate = sf.read('talk.wav')
fig = plt.figure()
plt.plot(data)
plt.title("Música")
fig.tight_layout()
plt.show()  

data, samplerate = sf.read('success.wav')
fig = plt.figure()
plt.plot(data)
plt.title("Música")
fig.tight_layout()
plt.show()  

f = sf.SoundFile('talk.wav')
print('Talk samples = {}'.format(f.frames))
print('sample rate = {}'.format(f.samplerate))
print('seconds = {}'.format(f.frames / f.samplerate))



f = sf.SoundFile('success.wav')
print('Success samples = {}'.format(f.frames))
print('sample rate = {}'.format(f.samplerate))
print('seconds = {}'.format(f.frames / f.samplerate))