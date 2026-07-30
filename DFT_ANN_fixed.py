from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

N = 32
batch = 10000

sig = np.random.randn(batch, N) + 1j*np.random.randn(batch, N)
F = np.fft.fft(sig, axis=-1)

X = np.hstack([sig.real, sig.imag])
Y = np.hstack([F.real, F.imag])

model = Sequential([Dense(N*2, input_dim=N*2, use_bias=False)])
model.compile(loss='mean_squared_error', optimizer='adam')

callbacks = [
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-8),
    EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
]

history = model.fit(X, Y, epochs=200, batch_size=100, callbacks=callbacks, verbose=1)

# --- Confirm it works ---
data = np.arange(N)

def ANN_DFT(x):
    if len(x) != N:
        raise ValueError(f'Input must be length {N}')
    pred = model.predict(np.hstack([x.real, x.imag])[np.newaxis])[0]
    return pred[:N] + 1j*pred[N:]

ANN = ANN_DFT(data)
FFT = np.fft.fft(data)
print(f'ANN matches FFT: {np.allclose(ANN, FFT)}')

# --- Weight heatmap ---
plt.figure()
plt.imshow(model.get_weights()[0], vmin=-1, vmax=1, cmap='coolwarm')
plt.title('Learned weight matrix')
plt.colorbar()

# --- Loss curve ---
plt.figure()
plt.semilogy(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Training loss')
plt.grid(True, which='both', alpha=0.3)

plt.show()
