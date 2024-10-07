import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import wavfile
from scipy.signal import chirp

def jwst_image_color_sonification(image_path, duration=30, sample_rate=44100):
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Normalize the image data
    normalized_img = img_array / 255.0

    # Create a time array
    t = np.linspace(0, duration, num=duration*sample_rate, endpoint=False)

    # Initialize the audio signal
    audio_signal = np.zeros_like(t)

    # Define frequency ranges for each color channel
    freq_ranges = {
        'red': (100, 500),    # Lower frequencies
        'green': (500, 1500), # Mid-range frequencies
        'blue': (1500, 5000)  # Higher frequencies
    }

    # Generate audio based on image data
    for i, color in enumerate(['red', 'green', 'blue']):
        channel_data = normalized_img[:, :, i]
        
        # Calculate the average intensity for each column
        column_avg = np.mean(channel_data, axis=0)
        
        # Map column index to time
        times = np.linspace(0, duration, num=len(column_avg))
        
        # Map intensities to frequencies within the color's range
        freq_min, freq_max = freq_ranges[color]
        frequencies = freq_min + column_avg * (freq_max - freq_min)
        
        # Generate chirp signal
        color_signal = chirp(t, f0=frequencies[0], f1=frequencies[-1], t1=duration, method='logarithmic')
        
        # Modulate amplitude based on overall intensity of the color channel
        intensity_envelope = np.interp(t, times, column_avg)
        modulated_signal = color_signal * intensity_envelope
        
        # Add to main audio signal
        audio_signal += modulated_signal

    # Normalize final audio signal
    audio_signal = audio_signal / np.max(np.abs(audio_signal))

    return audio_signal, sample_rate

# Example usage
image_path = 'ronadlo.png'
audio, sr = jwst_image_color_sonification(image_path)

# Save as WAV file
wavfile.write('jwst_color_sonification.wav', sr, (audio * 32767).astype(np.int16))

# Plot the audio waveform
plt.figure(figsize=(12, 4))
plt.plot(audio)
plt.title('JWST Image Color Sonification Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# Optional: Plot the spectrogram
plt.figure(figsize=(12, 8))
plt.specgram(audio, Fs=sr, scale='dB', cmap='inferno')
plt.title('JWST Image Color Sonification Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(label='Intensity (dB)')
plt.show()