import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import time
n = 8
omega = 1100
N = 256

# Generating random signal.

x = np.zeros(N)

for i in range(1, n+1):
    A = np.random.random()
    phi = np.random.random()

    for t in range(N):
        x[t] += A * np.sin(omega/i * (t + 1) + phi)
plt.figure(figsize=(14, 8))
plt.title("Pseudo-random variable")
plt.plot(range(N), x, "r")  # random variable
plt.show()


# Simple DFT. (for comparison)
def discrete_fourier_transform(signal):
    signal = np.asarray(signal, dtype=float)
    N = signal.shape[0]

    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, signal)


# Fast DFT.
def fast_fourier_transform(signal):
    signal = np.asarray(signal, dtype=float)
    N = signal.shape[0]

    if N <= 2:
        return discrete_fourier_transform(signal)
    else:
        signal_even = fast_fourier_transform(signal[::2])
        signal_odd = fast_fourier_transform(signal[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([signal_even + terms[:int(N / 2)] * signal_odd,
                               signal_even + terms[int(N / 2):] * signal_odd])


# Comparing DFT implementation with FFT implementation.
DFT = discrete_fourier_transform(x)
DFT_R = DFT.real
DFT_I = DFT.imag

FFT = fast_fourier_transform(x)
FFT_R = FFT.real
FFT_I = FFT.imag
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14,14))
ax1.plot(range(N), DFT_R)
ax2.plot(range(N), DFT_I)
ax3.plot(range(N), FFT_R)
ax4.plot(range(N), FFT_I)
plt.show()