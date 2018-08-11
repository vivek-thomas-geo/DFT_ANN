"""
Train a neural network to implement the discrete Fourier transform
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

N = 32
batch = 10000

# Generate random input data and desired output data
sig = np.random.randn(batch, N) + 1j*np.random.randn(batch, N)
F = fftpack.fft(sig, axis=-1)

# First half of inputs/outputs is real part, second half is imaginary part
X = np.hstack([sig.real, sig.imag])
Y = np.hstack([F.real, F.imag])

# Create model with no hidden layers, same number of outputs as inputs.
# No bias needed.  No activation function, since DFT is linear.
model = Sequential([Dense(N*2, input_dim=N*2, use_bias=False)])
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=100)

# Confirm that it works
data = np.arange(32)
pre = model.predict(np.array([np.concatenate((data.real, data.imag))]))[0]
ANN = pre[:32] + 1j*pre[32:]
FFT = np.fft.fft(data)
print(f'ANN matches FFT: {np.allclose(ANN, FFT)}')

# Heat map of neuron weights
plt.imshow(model.get_weights()[0], vmin=-1, vmax=1, cmap='coolwarm')
