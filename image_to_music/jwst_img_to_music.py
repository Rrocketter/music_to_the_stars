import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import wavfile
from scipy.signal import chirp, sawtooth, square
from scipy.stats import entropy

def analyze_image(img_array):
    """Analyze image characteristics to determine sonification parameters."""
    # Normalize image
    img_normalized = img_array / 255.0
    
    # Calculate overall brightness and contrast
    brightness = np.mean(img_normalized)
    contrast = np.std(img_normalized)
    
    # Calculate color channel distributions
    color_distribution = np.mean(img_normalized, axis=(0,1))
    
    # Calculate image complexity using entropy
    complexity = np.mean([entropy(img_normalized[:,:,i].flatten()) for i in range(3)])
    
    # Calculate spatial frequency (rough measure of detail level)
    spatial_freq = np.mean([np.abs(np.diff(img_normalized[:,:,i])).mean() for i in range(3)])
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'color_distribution': color_distribution,
        'complexity': complexity,
        'spatial_frequency': spatial_freq
    }

def get_adaptive_parameters(image_stats):
    """Generate sonification parameters based on image characteristics."""
    # BPM variation (80-160 BPM)
    bpm = 80 + int(image_stats['complexity'] * 80)
    
    # Beat intensity (0.2-0.5)
    beat_intensity = 0.2 + image_stats['contrast'] * 0.1
    
    # Base frequency (100-200 Hz)
    base_freq = 100 + image_stats['brightness'] * 100
    
    # Number of harmonics (1-5, based on complexity)
    num_harmonics = max(1, min(5, int(image_stats['spatial_frequency'] * 10)))
    
    # Harmonic intensities based on color distribution
    harmonic_intensities = [0.15 * (1 / (i + 1)) * (1 + image_stats['contrast']) 
                           for i in range(num_harmonics)]
    
    # Frequency ranges influenced by color distribution
    freq_multipliers = {
        'red': 1 + image_stats['color_distribution'][0],
        'green': 1.5 + image_stats['color_distribution'][1],
        'blue': 2 + image_stats['color_distribution'][2]
    }
    
    return {
        'bpm': bpm,
        'beat_intensity': beat_intensity,
        'base_freq': base_freq,
        'num_harmonics': num_harmonics,
        'harmonic_intensities': harmonic_intensities,
        'freq_multipliers': freq_multipliers
    }

def generate_waveform(t, freq, wave_type='sine'):
    """Generate different waveform types."""
    if wave_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        return square(2 * np.pi * freq * t)
    elif wave_type == 'sawtooth':
        return sawtooth(2 * np.pi * freq * t)
    return np.sin(2 * np.pi * freq * t)

def generate_beat(duration, sample_rate, bpm, complexity):
    """Generate a rhythmic beat with variations based on image complexity."""
    beat_interval = sample_rate * 60 / bpm
    num_samples = int(duration * sample_rate)
    beat = np.zeros(num_samples)
    
    # Generate kick drum
    for i in range(0, num_samples, int(beat_interval)):
        if i + 1000 < num_samples:
            t = np.linspace(0, 0.1, 1000)
            kick = np.sin(2 * np.pi * 100 * np.exp(-10 * t))
            beat[i:i+1000] += kick * np.exp(-10 * t)
    
    # Add variation based on complexity
    if complexity > 0.5:  # Add syncopated beats for complex images
        for i in range(int(beat_interval/2), num_samples, int(beat_interval)):
            if i + 500 < num_samples:
                t = np.linspace(0, 0.05, 500)
                syncopation = np.sin(2 * np.pi * 200 * np.exp(-20 * t)) * 0.5
                beat[i:i+500] += syncopation * np.exp(-20 * t)
    
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

def jwst_image_color_sonification(image_path, duration=30, sample_rate=44100):
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Analyze image and get adaptive parameters
    image_stats = analyze_image(img_array)
    params = get_adaptive_parameters(image_stats)
    
    # Create time array
    t = np.linspace(0, duration, num=duration*sample_rate, endpoint=False)
    
    # Initialize audio signal
    audio_signal = np.zeros_like(t)
    
    # Generate beat with adaptive BPM
    beat = generate_beat(duration, sample_rate, params['bpm'], image_stats['complexity'])
    
    # Process each color channel
    for color in ['red', 'green', 'blue']:
        channel_idx = ['red', 'green', 'blue'].index(color)
        channel_data = img_array[:, :, channel_idx] / 255.0
        column_avg = np.mean(channel_data, axis=0)
        
        # Calculate frequency range for this color
        base_freq = params['base_freq'] * params['freq_multipliers'][color]
        freq_range = (base_freq, base_freq * 2)
        
        # Generate and process audio for this channel
        times = np.linspace(0, duration, num=len(column_avg))
        frequencies = freq_range[0] + column_avg * (freq_range[1] - freq_range[0])
        
        main_signal = np.zeros_like(t)
        for time_idx, freq in enumerate(frequencies):
            time_start = int((time_idx / len(frequencies)) * len(t))
            time_end = int(((time_idx + 1) / len(frequencies)) * len(t))
            
            # Generate main tone
            wave = generate_waveform(t[time_start:time_end], freq)
            
            # Add harmonics
            for h in range(params['num_harmonics']):
                harmonic_freq = freq * (h + 2)
                if harmonic_freq < 2000:  # Frequency limiting
                    wave += params['harmonic_intensities'][h] * generate_waveform(
                        t[time_start:time_end], harmonic_freq)
            
            main_signal[time_start:time_end] = wave
        
        # Apply envelope and intensity modulation
        intensity_envelope = np.interp(t, times, column_avg)
        modulated_signal = apply_envelope(main_signal) * intensity_envelope
        
        # Add to main audio signal
        audio_signal += modulated_signal * (0.2 * image_stats['color_distribution'][channel_idx])
    
    # Add beat with adaptive intensity
    audio_signal += beat * params['beat_intensity']
    
    # Normalize and apply gentle low-pass filter
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 2000/(sample_rate/2), btype='low')
    audio_signal = filtfilt(b, a, audio_signal)
    
    return audio_signal, sample_rate, params

# Example usage
image_path = '5.jpg'
audio, sr, parameters = jwst_image_color_sonification(image_path)

# Print the adaptive parameters for this image
print("\nAdaptive parameters for this image:")
print(f"BPM: {parameters['bpm']}")
print(f"Beat Intensity: {parameters['beat_intensity']:.2f}")
print(f"Base Frequency: {parameters['base_freq']:.1f} Hz")
print(f"Number of Harmonics: {parameters['num_harmonics']}")

# Save as WAV file
wavfile.write('5.wav', sr, (audio * 32767).astype(np.int16))

# Plot the audio waveform
plt.figure(figsize=(12, 4))
plt.plot(audio)
plt.title('Adaptive JWST Image Sonification Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# Plot the spectrogram
plt.figure(figsize=(12, 8))
plt.specgram(audio, Fs=sr, scale='dB', cmap='inferno')
plt.title('Adaptive JWST Image Sonification Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(label='Intensity (dB)')
plt.show()