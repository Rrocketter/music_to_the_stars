import numpy as np
from astropy.io import fits
from scipy.io import wavfile
from scipy.signal import chirp, sawtooth, square, butter, filtfilt
from scipy.stats import entropy
import matplotlib.pyplot as plt

def generate_waveform(t, freq, wave_type='sine', width=0.5):
    if wave_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        return square(2 * np.pi * freq * t, duty=width)
    elif wave_type == 'sawtooth':
        return sawtooth(2 * np.pi * freq * t)
    elif wave_type == 'triangle':
        return sawtooth(2 * np.pi * freq * t, width=0.5)
    return np.sin(2 * np.pi * freq * t)

def apply_envelope(signal, attack=0.1, decay=0.2, sustain=0.7, release=0.2, sustain_level=0.7):
    total_length = len(signal)
    attack_len = int(attack * total_length)
    decay_len = int(decay * total_length)
    release_len = int(release * total_length)
    sustain_len = total_length - attack_len - decay_len - release_len
    
    envelope = np.zeros(total_length)
    envelope[:attack_len] = np.linspace(0, 1, attack_len)
    envelope[attack_len:attack_len+decay_len] = np.linspace(1, sustain_level, decay_len)
    envelope[attack_len+decay_len:attack_len+decay_len+sustain_len] = sustain_level
    envelope[-release_len:] = np.linspace(sustain_level, 0, release_len)
    
    return signal * envelope

def generate_beat(duration, sample_rate, bpm, complexity):
    beat_interval = sample_rate * 60 / bpm
    num_samples = int(duration * sample_rate)
    beat = np.zeros(num_samples)
    
    for i in range(0, num_samples, int(beat_interval)):
        if i + 1000 < num_samples:
            t = np.linspace(0, 0.1, 1000)
            kick = np.sin(2 * np.pi * 100 * np.exp(-10 * t))
            beat[i:i+1000] += kick * np.exp(-10 * t)
    
    if complexity > 0.3:
        for i in range(int(beat_interval/2), num_samples, int(beat_interval)):
            if i + 500 < num_samples:
                t = np.linspace(0, 0.05, 500)
                offbeat = np.sin(2 * np.pi * 200 * np.exp(-20 * t)) * 0.5
                beat[i:i+500] += offbeat * np.exp(-20 * t)
    
    if complexity > 0.6:
        sixteenth = int(beat_interval/4)
        for i in range(sixteenth, num_samples, sixteenth):
            if i + 200 < num_samples and np.random.random() > 0.5:
                t = np.linspace(0, 0.02, 200)
                hi_hat = np.random.random(200) * np.exp(-30 * t) * 0.3
                beat[i:i+200] += hi_hat
    
    beat = beat / np.max(np.abs(beat))
    return beat

def analyze_fits_data(data):
    if data is None or not isinstance(data, np.ndarray):
        raise ValueError("Invalid FITS data. Expected a numpy array, got {}".format(type(data)))
    
    if np.all(data == 0):
        raise ValueError("FITS data is all zeros. Please check the FITS file.")
    
    # Normalize the data
    normalized_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    
    # Replace NaN values with 0 for further calculations
    normalized_data = np.nan_to_num(normalized_data)
    
    brightness = np.mean(normalized_data)
    contrast = np.std(normalized_data)
    complexity = entropy(normalized_data.flatten())
    spatial_freq = np.mean([np.abs(np.diff(normalized_data, axis=i)).mean() for i in range(normalized_data.ndim)])
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'complexity': complexity,
        'spatial_frequency': spatial_freq,
    }

def get_adaptive_parameters(fits_stats):
    base_bpm = 80
    bpm_range = 80
    bpm = base_bpm + int(fits_stats['complexity'] * bpm_range)
    bpm = np.clip(bpm, 80, 160)
    
    beat_intensity = 0.2 + (fits_stats['contrast'] * 0.3)
    beat_intensity = np.clip(beat_intensity, 0.2, 0.5)

    base_freq = 100 + (fits_stats['brightness'] * 300)
    base_freq = np.clip(base_freq, 100, 400)
    
    num_harmonics = int(5 + fits_stats['spatial_frequency'] * 5)
    num_harmonics = np.clip(num_harmonics, 1, 10)

    harmonic_intensities = [np.random.uniform(0.1, 1.0) for _ in range(num_harmonics)]
    
    wave_types = ['sine', 'triangle', 'sawtooth', 'square']
    wave_type = wave_types[int(fits_stats['complexity'] * len(wave_types)) % len(wave_types)]
    
    envelope_params = {
        'attack': 0.1 + fits_stats['spatial_frequency'] * 0.1,
        'decay': 0.2 + fits_stats['complexity'] * 0.1,
        'sustain': 0.7 - fits_stats['contrast'] * 0.2,
        'release': 0.2 + fits_stats['brightness'] * 0.1
    }
    
    return {
        'bpm': bpm,
        'beat_intensity': beat_intensity,
        'base_freq': base_freq,
        'num_harmonics': num_harmonics,
        'harmonic_intensities': harmonic_intensities,
        'wave_type': wave_type,
        'envelope_params': envelope_params
    }

def jwst_fits_to_sound(file_path, duration=30, sample_rate=44100):
    try:
        # Load the FITS file
        with fits.open(file_path) as hdul:
            print("FITS file structure:")
            hdul.info()
            
            # Try to get the data from the SCI extension
            fits_data = hdul['SCI'].data
            
            if fits_data is None:
                raise ValueError("No data found in the SCI extension of the FITS file.")
            
            print(f"FITS data shape: {fits_data.shape}")
            print(f"FITS data type: {fits_data.dtype}")
            print(f"Min value: {np.nanmin(fits_data)}, Max value: {np.nanmax(fits_data)}")
    
    except Exception as e:
        print(f"Error reading FITS file: {e}")
        return None, None, None

    # Analyze FITS data and get adaptive parameters
    try:
        fits_stats = analyze_fits_data(fits_data)
        params = get_adaptive_parameters(fits_stats)
    except Exception as e:
        print(f"Error analyzing FITS data: {e}")
        return None, None, None

    # Create a time array
    t = np.linspace(0, duration, num=duration*sample_rate, endpoint=False)
    
    # Initialize the audio signal
    audio_signal = np.zeros_like(t)
    
    # Generate beat
    beat = generate_beat(duration, sample_rate, params['bpm'], fits_stats['complexity'])
    
    # Generate audio based on FITS data
    num_rows = fits_data.shape[0]
    for i, row in enumerate(fits_data):
        # Map row index to frequency (higher rows = higher frequency)
        frequency = params['base_freq'] + (i / num_rows) * params['base_freq']
        
        # Generate main wave for this row
        row_signal = generate_waveform(t, frequency, params['wave_type'])
        
        # Add harmonics
        for h in range(params['num_harmonics']):
            harmonic_freq = frequency * (h + 2)
            if harmonic_freq < sample_rate / 2:  # Nyquist frequency
                row_signal += params['harmonic_intensities'][h] * generate_waveform(t, harmonic_freq, params['wave_type'])
        
        # Modulate amplitude based on pixel values
        intensity_envelope = np.interp(t, np.linspace(0, duration, num=len(row)), np.nan_to_num(row))
        modulated_signal = apply_envelope(row_signal, **params['envelope_params']) * intensity_envelope
        
        # Add to main audio signal
        audio_signal += modulated_signal / num_rows  # Normalize by number of rows
    
    # Mix beat and main signal
    audio_signal = audio_signal * 0.7 + beat * params['beat_intensity']
    
    # Normalize final audio signal
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    
    # Apply gentle low-pass filter
    b, a = butter(4, 2000/(sample_rate/2), btype='low')
    audio_signal = filtfilt(b, a, audio_signal)
    
    return audio_signal, sample_rate, params

# Example usage
file_path = 'sample.fits'
audio, sr, parameters = jwst_fits_to_sound(file_path)

if audio is not None:
    # Save as WAV file
    wavfile.write('jwst_sonification.wav', sr, (audio * 32767).astype(np.int16))

    # Plot the audio waveform
    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    plt.title('JWST FITS Data Sonification Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

    # Print out the adaptive parameters
    print("\nAdaptive parameters for this FITS data:")
    print(f"BPM: {parameters['bpm']}")
    print(f"Beat Intensity: {parameters['beat_intensity']:.2f}")
    print(f"Base Frequency: {parameters['base_freq']:.1f} Hz")
    print(f"Number of Harmonics: {parameters['num_harmonics']}")
    print(f"Wave Type: {parameters['wave_type']}")
else:
    print("Failed to generate audio from the FITS file.")