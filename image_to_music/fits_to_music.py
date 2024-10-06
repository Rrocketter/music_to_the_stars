import numpy as np
from astropy.io import fits
from scipy.io import wavfile
import matplotlib.pyplot as plt

def jwst_image_to_sound(file_path, duration=30, sample_rate=44100):
    # Load the FITS file
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data

    # Normalize the data
    normalized_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # Create a time array
    t = np.linspace(0, duration, num=duration*sample_rate, endpoint=False)

    # Initialize the audio signal
    audio_signal = np.zeros_like(t)

    # Generate audio based on image data
    for i, row in enumerate(normalized_data):
        # Map row index to frequency (higher rows = higher frequency)
        frequency = 100 + (i / len(normalized_data)) * 1000
        
        # Generate sine wave for this row
        row_signal = np.sin(2 * np.pi * frequency * t)
        
        # Modulate amplitude based on pixel values
        modulated_signal = row_signal * np.interp(t, np.linspace(0, duration, num=len(row)), row)
        
        # Add to main audio signal
        audio_signal += modulated_signal

    # Normalize final audio signal
    audio_signal = audio_signal / np.max(np.abs(audio_signal))

    return audio_signal, sample_rate

# Example usage
file_path = 'path_to_your_jwst_fits_file.fits'
audio, sr = jwst_image_to_sound(file_path)

# Save as WAV file
wavfile.write('jwst_sonification.wav', sr, (audio * 32767).astype(np.int16))

# Plot the audio waveform
plt.figure(figsize=(12, 4))
plt.plot(audio)
plt.title('JWST Image Sonification Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()