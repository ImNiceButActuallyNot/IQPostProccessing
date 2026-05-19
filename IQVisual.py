
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Path to your .mat file; update this to your actual filename
mat_file = r"C:\Users\boaz.arnon\Downloads\iq_2430MHz_20260316_155558_870215\iq_2430MHz_20260316_155558_870215.mat"
 
#r"Z:\SVI Team\Team\Boaz\Real Time Spectrum Files\IQ recordings\Zebra Printer\16.3.2026 - iteration 6 (no ch38)_files\mat_files\iq_2430MHz_20260316_171035_136879.mat"


# Load the .mat file
data = loadmat(mat_file)

# Extract the complex IQ data
Y = data['Y'].squeeze()
fc = data['InputCenter'][0][0]  # Center frequency

# Determine time step (XDelta) if available, otherwise default to 1
if 'XDelta' in data:
    dt = float(np.squeeze(data['XDelta']))
else:
    dt = 1.0

# Time axis
t = np.arange(Y.size) * dt

# Compute FFT and frequency axis
N = Y.size
# Subtract mean to remove DC offset (LO leakage artifact at center frequency)
Y_fft = np.fft.fftshift(np.fft.fft(Y - np.mean(Y)))
freqs = np.fft.fftshift(np.fft.fftfreq(N, dt)) + fc  # sorted: fc-fs/2 .. fc+fs/2

# Only frequencies in desired range
pos = (freqs >= 2.4e9) & (freqs <= 2.48e9)
freqs_pos = freqs[pos]

# --------- SMOOTHED SPECTRUM & -3 dB BW ---------
# Magnitude spectrum
spec = np.abs(Y_fft[pos])
# Convert to dB (relative to max)
spec_db = 20 * np.log10(spec + 1e-12)  # add epsilon to avoid log(0)

# Simple moving-average smoothing
win_len = 40  # adjust as you like (must be odd for symmetric smoothing)
kernel = np.ones(win_len) / win_len
spec_db_smooth = np.convolve(spec_db, kernel, mode='same')

# Find -10 dB bandwidth on smoothed spectrum
peak_db = np.max(spec_db_smooth)
thresh_db = peak_db - 10.0

above = spec_db_smooth >= thresh_db
idx = np.where(above)[0]

bw_3db = None
if idx.size >= 2:
    f_low = freqs_pos[idx[0]]
    f_high = freqs_pos[idx[-1]]
    bw_10db = f_high - f_low
    print(f"-10 dB BW: {bw_10db/1e6:.3f} MHz")
else:
    print("Could not find a contiguous region above -10 dB threshold.")

# --------- PLOTS ---------
plt.figure(figsize=(10, 8))

# Time-domain magnitude |x(t)|
plt.subplot(4, 1, 1)
plt.plot(1000 * t, 10 * np.log(np.abs(Y)))
plt.xlabel('Time (msec)')
plt.ylabel('|x(t)|')
plt.title('Time-Domain IQ Magnitude')

# Frequency-domain magnitude |X(f)| (unsmoothed)
plt.subplot(4, 1, 2)
plt.plot(freqs_pos, 10 * np.log(np.abs(Y_fft[pos])))
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X(f)|')
plt.title('Frequency-Domain IQ Magnitude (Raw)')

# Smoothed spectrum in dB with -10 dB line
plt.subplot(4, 1, 3)
plt.plot(freqs_pos, spec_db_smooth, label='Smoothed |X(f)| (dB)')
plt.axhline(thresh_db, linestyle='--', label='-10 dB level')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
title_str = 'Smoothed Spectrum'
if bw_10db is not None:
    title_str += f' (BW ≈ {bw_10db/1e6:.3f} MHz)'
plt.title(title_str)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Center Frequency: {fc/1e6:.3f} MHz")

# %%
