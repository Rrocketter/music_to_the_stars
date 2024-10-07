import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import wavfile
from scipy.signal import chirp, sawtooth, square

def generate_waveform(t, freq, wave_type='sine'):
    """Generate different waveform types."""
    if wave_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        return square(2 * np.pi * freq * t)
    elif wave_type == 'sawtooth':
        return sawtooth(2 * np.pi * freq * t)
    return np.sin(2 * np.pi * freq * t)

def generate_beat(duration, sample_rate, bpm=120):
    """Generate a basic rhythmic beat."""
    beat_interval = sample_rate * 60 / bpm  # samples between beats
    num_samples = int(duration * sample_rate)
    beat = np.zeros(num_samples)
    
    # Generate kick drum sound
    for i in range(0, num_samples, int(beat_interval)):
        if i + 1000 < num_samples:
            t = np.linspace(0, 0.1, 1000)
            kick = np.sin(2 * np.pi * 100 * np.exp(-10 * t))
            beat[i:i+1000] += kick * np.exp(-10 * t)
    
    return beat

def apply_envelope(signal, attack=0.1, decay=0.2, sustain=0.7, release=0.2):
    """Apply ADSR envelope to a signal."""
    total_length = len(signal)
    attack_len = int(attack * total_length)
    decay_len = int(decay * total_length)
    release_len = int(release * total_length)
    sustain_len = total_length - attack_len - decay_len - release_len
    
    envelope = np.zeros(total_length)
    envelope[:attack_len] = np.linspace(0, 1, attack_len)
    envelope[attack_len:attack_len+decay_len] = np.linspace(1, sustain, decay_len)
    envelope[attack_len+decay_len:attack_len+decay_len+sustain_len] = sustain
    envelope[-release_len:] = np.linspace(sustain, 0, release_len)
    
    return signal * envelope

def limit_frequency(freq, max_freq=2000):
    """Limit frequency to prevent harsh high frequencies."""
    return min(freq, max_freq)

def jwst_image_color_sonification(image_path, duration=30, sample_rate=44100):
    # Load and normalize image
    img = Image.open(image_path)
    img_array = np.array(img)
    normalized_img = img_array / 255.0
    
    # Create time array
    t = np.linspace(0, duration, num=duration*sample_rate, endpoint=False)
    
    # Initialize audio signal
    audio_signal = np.zeros_like(t)
    
    # Define musical scale (pentatonic scale frequencies)
    base_freq = 110  # A2 note (lower base frequency)
    scale_multipliers = [1, 1.125, 1.25, 1.5, 1.666]
    
    # Define frequency ranges for each color channel (lowered ranges)
    freq_ranges = {
        'red': (base_freq, base_freq * 1.5),      # 110 Hz - 165 Hz
        'green': (base_freq * 1.5, base_freq * 2), # 165 Hz - 220 Hz
        'blue': (base_freq * 2, base_freq * 3)     # 220 Hz - 330 Hz
    }
    
    # Different waveforms for each color
    waveforms = {
        'red': 'sine',
        'green': 'sine',  # Changed from square to reduce harshness
        'blue': 'sine'    # Changed from sawtooth to reduce harshness
    }
    
    # Generate beat
    beat = generate_beat(duration, sample_rate, bpm=120)
    
    # Generate audio based on image data
    for i, color in enumerate(['red', 'green', 'blue']):
        channel_data = normalized_img[:, :, i]
        column_avg = np.mean(channel_data, axis=0)
        
        times = np.linspace(0, duration, num=len(column_avg))
        freq_min, freq_max = freq_ranges[color]
        
        frequencies = freq_min + column_avg * (freq_max - freq_min)
        frequencies = np.array([limit_frequency(f) for f in frequencies])
        
        # Quantize frequencies to pentatonic scale
        scale_frequencies = np.array([freq_min * m for m in scale_multipliers])
        frequencies = np.array([scale_frequencies[np.argmin(np.abs(scale_frequencies - f))] 
                              for f in frequencies])
        
        main_signal = np.zeros_like(t)
        for time_idx, freq in enumerate(frequencies):
            time_start = int((time_idx / len(frequencies)) * len(t))
            time_end = int(((time_idx + 1) / len(frequencies)) * len(t))
            
            wave = generate_waveform(t[time_start:time_end], 
                                   freq, 
                                   wave_type=waveforms[color])
            
            # Reduced number of harmonics and their intensity
            for harmonic in [2]:  # Only one harmonic
                wave += 0.15 / harmonic * generate_waveform(t[time_start:time_end], 
                                                          freq * harmonic, 
                                                          wave_type=waveforms[color])
            
            main_signal[time_start:time_end] = wave
        
        intensity_envelope = np.interp(t, times, column_avg)
        modulated_signal = apply_envelope(main_signal) * intensity_envelope
        
        # Reduced channel weights
        audio_signal += modulated_signal * (0.2 if color == 'green' else 0.15)
    
    # Add beat to the final mix
    audio_signal += beat * 0.3
    
    # Normalize final audio signal
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    
    # Apply a gentle low-pass filter
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 2000/(sample_rate/2), btype='low')
    audio_signal = filtfilt(b, a, audio_signal)
    
    return audio_signal, sample_rate

# Example usage
image_path = '2.jpg'
audio, sr = jwst_image_color_sonification(image_path)

# Save as WAV file
wavfile.write('jwst_color_sonification_with_beat.wav', sr, (audio * 32767).astype(np.int16))

# Plot the audio waveform
plt.figure(figsize=(12, 4))
plt.plot(audio)
plt.title('Enhanced JWST Image Color Sonification Waveform with Beat')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# Plot the spectrogram
plt.figure(figsize=(12, 8))
plt.specgram(audio, Fs=sr, scale='dB', cmap='inferno')
plt.title('Enhanced JWST Image Color Sonification Spectrogram with Beat')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(label='Intensity (dB)')
plt.show()