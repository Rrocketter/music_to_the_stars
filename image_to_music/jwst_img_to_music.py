import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import wavfile
from scipy.signal import chirp, sawtooth, square, butter, filtfilt
from scipy.stats import entropy
import random

def generate_waveform(t, freq, wave_type='sine', width=0.5):
    """
    Generate different waveform types.
    
    Args:
        t: Time array
        freq: Frequency in Hz
        wave_type: 'sine', 'square', 'sawtooth', or 'triangle'
        width: Pulse width for square wave
    """
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
    """
    Apply ADSR (Attack, Decay, Sustain, Release) envelope to a signal.
    
    Args:
        signal: Input audio signal
        attack: Attack time as fraction of total duration
        decay: Decay time as fraction of total duration
        sustain: Sustain time as fraction of total duration
        release: Release time as fraction of total duration
        sustain_level: Amplitude level during sustain phase (0-1)
    """
    total_length = len(signal)
    attack_len = int(attack * total_length)
    decay_len = int(decay * total_length)
    release_len = int(release * total_length)
    sustain_len = total_length - attack_len - decay_len - release_len
    
    envelope = np.zeros(total_length)
    
    # Attack phase
    envelope[:attack_len] = np.linspace(0, 1, attack_len)
    
    # Decay phase
    envelope[attack_len:attack_len+decay_len] = np.linspace(1, sustain_level, decay_len)
    
    # Sustain phase
    envelope[attack_len+decay_len:attack_len+decay_len+sustain_len] = sustain_level
    
    # Release phase
    envelope[-release_len:] = np.linspace(sustain_level, 0, release_len)
    
    return signal * envelope

def generate_beat(duration, sample_rate, bpm, complexity):
    """
    Generate a rhythmic beat pattern based on image complexity.
    
    Args:
        duration: Total duration in seconds
        sample_rate: Audio sample rate
        bpm: Beats per minute
        complexity: Image complexity measure (0-1)
    """
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

def get_adaptive_parameters(image_stats):
    """
    Generate sonification parameters based on image characteristics.
    
    Args:
        image_stats: Dictionary containing image analysis results
    """
    base_bpm = 80
    bpm_range = 80
    bpm = base_bpm + int((image_stats['complexity'] * 0.7 + 
                         image_stats['edge_intensity'] * 0.3) * bpm_range)
    bpm = np.clip(bpm, 80, 160)
    
    beat_intensity = 0.2 + (image_stats['contrast'] * 0.3 + 
                           image_stats['edge_intensity'] * 0.2)
    beat_intensity = np.clip(beat_intensity, 0.2, 0.5)

    base_freq = 100 + (image_stats['brightness'] * 80 + 
                      image_stats['color_variance'] * 20)
    base_freq = np.clip(base_freq, 100, 200)
    
    num_harmonics = random.randint(1, 10)  # Adjust the range as needed

    harmonic_intensities = [random.uniform(0.1, 1.0) for i in range(num_harmonics)]
    
    freq_multipliers = {
        'red': 1 + image_stats['color_distribution'][0],
        'green': 1.5 + image_stats['color_distribution'][1],
        'blue': 2 + image_stats['color_distribution'][2]
    }
    
    wave_types = {
        'red': 'sine',
        'green': 'triangle' if image_stats['complexity'] > 0.5 else 'sine',
        'blue': 'sawtooth' if image_stats['edge_intensity'] > 0.6 else 'sine'
    }
    
    # ADSR envelope parameters based on image characteristics
    envelope_params = {
        'attack': 0.1 + image_stats['edge_intensity'] * 0.1,
        'decay': 0.2 + image_stats['complexity'] * 0.1,
        'sustain': 0.7 - image_stats['contrast'] * 0.2,
        'release': 0.2 + image_stats['brightness'] * 0.1
    }
    
    return {
        'bpm': bpm,
        'beat_intensity': beat_intensity,
        'base_freq': base_freq,
        'num_harmonics': num_harmonics,
        'harmonic_intensities': harmonic_intensities,
        'freq_multipliers': freq_multipliers,
        'wave_types': wave_types,
        'envelope_params': envelope_params
    }



def get_chord_frequencies(root_freq, chord_type='major'):
    """Generate frequencies for a chord based on root frequency."""
    if chord_type == 'major':
        return [root_freq, root_freq * 1.25, root_freq * 1.5]  # Major triad
    elif chord_type == 'minor':
        return [root_freq, root_freq * 1.2, root_freq * 1.5]   # Minor triad
    elif chord_type == 'diminished':
        return [root_freq, root_freq * 1.2, root_freq * 1.4]   # Diminished triad
    elif chord_type == 'augmented':
        return [root_freq, root_freq * 1.25, root_freq * 1.6]  # Augmented triad
    return [root_freq, root_freq * 1.25, root_freq * 1.5]      # Default to major

def generate_chord_progression(image_stats, duration, sample_rate):
    """Generate chord progression based on image characteristics."""
    # Determine chord progression complexity based on image
    complexity = image_stats['complexity']
    brightness = image_stats['brightness']
    contrast = image_stats['contrast']
    
    # Base chord progressions (from simple to complex)
    progressions = {
        'simple': ['I', 'IV', 'V', 'I'],
        'medium': ['I', 'vi', 'IV', 'V'],
        'complex': ['I', 'vi', 'ii', 'V', 'I', 'IV', 'V', 'I']
    }
    
    # Choose progression based on complexity
    if complexity < 0.3:
        progression = progressions['simple']
    elif complexity < 0.6:
        progression = progressions['medium']
    else:
        progression = progressions['complex']
    
    # Determine base frequency and chord types based on image brightness
    base_freq = 110 * (1 + brightness)  # Higher brightness = higher base frequency
    
    # Map chords to frequencies and types
    chord_map = {
        'I': (base_freq, 'major'),
        'ii': (base_freq * 9/8, 'minor'),
        'iii': (base_freq * 5/4, 'minor'),
        'IV': (base_freq * 4/3, 'major'),
        'V': (base_freq * 3/2, 'major'),
        'vi': (base_freq * 5/3, 'minor'),
        'vii': (base_freq * 15/8, 'diminished')
    }
    
    # Generate the chord sequence
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    chord_signal = np.zeros_like(t)
    
    # Duration for each chord
    chord_duration = duration / len(progression)
    samples_per_chord = int(sample_rate * chord_duration)
    
    # Generate each chord in the progression
    for i, chord in enumerate(progression):
        start_idx = i * samples_per_chord
        end_idx = start_idx + samples_per_chord
        
        # Get chord frequencies and type
        root_freq, chord_type = chord_map[chord]
        frequencies = get_chord_frequencies(root_freq, chord_type)
        
        # Generate chord tones
        chord_wave = np.zeros(samples_per_chord)
        for freq in frequencies:
            t_chord = np.linspace(0, chord_duration, samples_per_chord, endpoint=False)
            # Use sine waves for cleaner chord sound
            chord_wave += np.sin(2 * np.pi * freq * t_chord) * 0.3
        
        # Apply envelope to chord
        envelope = np.ones(samples_per_chord)
        attack = int(0.1 * samples_per_chord)
        release = int(0.2 * samples_per_chord)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        
        chord_signal[start_idx:end_idx] = chord_wave * envelope
    
    # Scale chord intensity based on contrast
    chord_intensity = 0.2 + (contrast * 0.3)
    return chord_signal * chord_intensity

def analyze_image(img_array):
    """Analyze image characteristics to determine sonification parameters."""
    # Previous analyze_image function content remains the same
    img_normalized = img_array / 255.0
    
    brightness = np.mean(img_normalized)
    contrast = np.std(img_normalized)
    color_distribution = np.mean(img_normalized, axis=(0,1))
    complexity = np.mean([entropy(img_normalized[:,:,i].flatten()) for i in range(3)])
    spatial_freq = np.mean([np.abs(np.diff(img_normalized[:,:,i])).mean() for i in range(3)])
    
    # Add harmonic analysis
    color_variance = np.var(color_distribution)
    edge_intensity = np.mean([np.abs(np.diff(img_normalized[:,:,i], axis=1)).mean() + 
                            np.abs(np.diff(img_normalized[:,:,i], axis=0)).mean() 
                            for i in range(3)])
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'color_distribution': color_distribution,
        'complexity': complexity,
        'spatial_frequency': spatial_freq,
        'color_variance': color_variance,
        'edge_intensity': edge_intensity
    }

def jwst_image_color_sonification(image_path, duration=30, sample_rate=44100):
    # Load and analyze image
    img = Image.open(image_path)
    img_array = np.array(img)
    image_stats = analyze_image(img_array)
    params = get_adaptive_parameters(image_stats)
    
    # Create time array
    t = np.linspace(0, duration, num=duration*sample_rate, endpoint=False)
    
    # Generate main components
    audio_signal = np.zeros_like(t)
    beat = generate_beat(duration, sample_rate, params['bpm'], image_stats['complexity'])
    chord_progression = generate_chord_progression(image_stats, duration, sample_rate)
    
    # Process each color channel (previous implementation remains the same)
    for color in ['red', 'green', 'blue']:
        channel_idx = ['red', 'green', 'blue'].index(color)
        channel_data = img_array[:, :, channel_idx] / 255.0
        column_avg = np.mean(channel_data, axis=0)
        
        base_freq = params['base_freq'] * params['freq_multipliers'][color]
        freq_range = (base_freq, base_freq * 2)
        
        times = np.linspace(0, duration, num=len(column_avg))
        frequencies = freq_range[0] + column_avg * (freq_range[1] - freq_range[0])
        
        main_signal = np.zeros_like(t)
        for time_idx, freq in enumerate(frequencies):
            time_start = int((time_idx / len(frequencies)) * len(t))
            time_end = int(((time_idx + 1) / len(frequencies)) * len(t))
            
            wave = generate_waveform(t[time_start:time_end], freq)
            
            for h in range(params['num_harmonics']):
                harmonic_freq = freq * (h + 2)
                if harmonic_freq < 2000:
                    wave += params['harmonic_intensities'][h] * generate_waveform(
                        t[time_start:time_end], harmonic_freq)
            
            main_signal[time_start:time_end] = wave
        
        intensity_envelope = np.interp(t, times, column_avg)
        modulated_signal = apply_envelope(main_signal) * intensity_envelope
        audio_signal += modulated_signal * (0.2 * image_stats['color_distribution'][channel_idx])
    
    # Mix all components
    audio_signal = audio_signal * 0.5  # Reduce main signal volume to make room for chords
    audio_signal += beat * params['beat_intensity']
    audio_signal += chord_progression  # Add chord progression
    
    # Normalize and apply gentle low-pass filter
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    b, a = butter(4, 2000/(sample_rate/2), btype='low')
    audio_signal = filtfilt(b, a, audio_signal)
    
    return audio_signal, sample_rate, params

image_path = '5.jpg'
audio, sr, parameters = jwst_image_color_sonification(image_path)

# # Example usage
# image_path = 'ronadlo.png'
# audio, sr = jwst_image_color_sonification(image_path)

print("\nAdaptive parameters for this image:")
print(f"BPM: {parameters['bpm']}")
print(f"Beat Intensity: {parameters['beat_intensity']:.2f}")
print(f"Base Frequency: {parameters['base_freq']:.1f} Hz")
print(f"Number of Harmonics: {parameters['num_harmonics']}")

wavfile.write('5.wav', sr, (audio * 32767).astype(np.int16))