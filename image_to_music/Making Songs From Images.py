
#Importing modules
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import IPython.display as ipd
import librosa
from midiutil import MIDIFile
import random
from pedalboard import Pedalboard, Chorus, Reverb
from pedalboard.io import AudioFile

#Load the image
ori_img = cv2.imread('colors.jpg')
img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

#Get shape of image
height, width, depth = img.shape
dpi = plt.rcParams['figure.dpi']
figsize = width / float(dpi), height / float(dpi)

#Plot the image
fig, axs = plt.subplots(1, 2, figsize = figsize)
axs[0].title.set_text('BGR') 
axs[0].imshow(ori_img)
axs[1].title.set_text('RGB') 
axs[1].imshow(img)
plt.show()
print('           Image Properties')
print('Height = ',height, 'Width = ', width)
print('Number of pixels in image = ', height * width)

#Need function that reads pixel hue value
hsv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
#Plot the image
fig, axs = plt.subplots(1, 3, figsize = (15,15))
names = ['BGR','RGB','HSV']
imgs  = [ori_img, img, hsv]
i = 0
for elem in imgs:
    axs[i].title.set_text(names[i])
    axs[i].imshow(elem)
    axs[i].grid(False)
    i += 1
plt.show()

i=0 ; j=0
#Initialize array the will contain Hues for every pixel in image
hues = [] 
for i in range(height):
    for j in range(width):
        hue = hsv[i][j][0] #This is the hue value at pixel coordinate (i,j)
        hues.append(hue)
pixels_df = pd.DataFrame(hues, columns=['hues'])
print('pixels_df', pixels_df)
#Define frequencies that make up A-Harmonic Minor Scale
scale_freqs = [220.00, 246.94 ,261.63, 293.66, 329.63, 349.23, 415.30] 
def hue2freq(h,scale_freqs):
    thresholds = [26 , 52 , 78 , 104,  128 , 154 , 180]
    note = scale_freqs[0]
    if (h <= thresholds[0]):
         note = scale_freqs[0]
    elif (h > thresholds[0]) & (h <= thresholds[1]):
        note = scale_freqs[1]
    elif (h > thresholds[1]) & (h <= thresholds[2]):
        note = scale_freqs[2]
    elif (h > thresholds[2]) & (h <= thresholds[3]):
        note = scale_freqs[3]
    elif (h > thresholds[3]) & (h <= thresholds[4]):    
        note = scale_freqs[4]
    elif (h > thresholds[4]) & (h <= thresholds[5]):
        note = scale_freqs[5]
    elif (h > thresholds[5]) & (h <= thresholds[6]):
        note = scale_freqs[6]
    else:
        note = scale_freqs[0]
    
    return note
pixels_df['notes'] = pixels_df.apply(lambda row : hue2freq(row['hues'],scale_freqs), axis = 1)
print('pixels_df 2', pixels_df)
frequencies = pixels_df['notes'].to_numpy()
song = np.array([])
sr = 22050 # sample rate
T = 0.1    # 0.1 second duration
t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
nPixels = 60
for i in range(nPixels):  
    val = frequencies[i]
    note  = 0.5*np.sin(2*np.pi*val*t)
    song  = np.concatenate([song, note])
ipd.Audio(song, rate=sr) # load a NumPy array
from scipy.io import wavfile
wavfile.write('ini_song.wav'    , rate = 22050, data = song.astype(np.float32))
song = np.array([])
octaves = np.array([0.5,1,2])
sr = 22050 # sample rate
T = 0.1    # 0.1 second duration
t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
nPixels = 60
for i in range(nPixels):
    octave = random.choice(octaves)
    val =  octave * frequencies[i]
    note  = 0.5*np.sin(2*np.pi*val*t)
    song  = np.concatenate([song, note])
ipd.Audio(song, rate=sr) # load a NumPy array
wavfile.write('octave_song.wav'    , rate = 22050, data = song.astype(np.float32))
song = np.array([])
octaves = np.array([1/2,1,2])
sr = 22050 # sample rate
T = 0.1    # 0.1 second duration
t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
nPixels = 60
for i in range(nPixels):
    octave = random.choice(octaves)
    val =  octave * random.choice(frequencies)
    note  = 0.5*np.sin(2*np.pi*val*t)
    song  = np.concatenate([song, note])
ipd.Audio(song, rate=sr) # load a NumPy array
wavfile.write('random_song.wav'    , rate = 22050, data = song.astype(np.float32))
def img2music(img, scale = [220.00, 246.94 ,261.63, 293.66, 329.63, 349.23, 415.30],
              sr = 22050, T = 0.1, nPixels = 60, useOctaves = True, randomPixels = False,
              harmonize = 'U0'):
    """
    Args:
        img    :     (array) image to process
        scale  :     (array) array containing frequencies to map H values to
        sr     :     (int) sample rate to use for resulting song
        T      :     (int) time in seconds for dutation of each note in song
        nPixels:     (int) how many pixels to use to make song
    Returns:
        song   :     (array) Numpy array of frequencies. Can be played by ipd.Audio(song, rate = sr)
    """
    #Convert image to HSV
    hsv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
    
    #Get shape of image
    height, width, depth = ori_img.shape

    i=0 ; j=0 ; k=0
    #Initialize array the will contain Hues for every pixel in image
    hues = [] 
    for i in range(height):
        for j in range(width):
            hue = hsv[i][j][0] #This is the hue value at pixel coordinate (i,j)
            hues.append(hue)
            
    #Make dataframe containing hues and frequencies
    pixels_df = pd.DataFrame(hues, columns=['hues'])
    pixels_df['frequencies'] = pixels_df.apply(lambda row : hue2freq(row['hues'],scale), axis = 1) 
    frequencies = pixels_df['frequencies'].to_numpy()
    
    #Make harmony dictionary (i.e. fundamental, perfect fifth, major third, octave)
    #unison           = U0 ; semitone         = ST ; major second     = M2
    #minor third      = m3 ; major third      = M3 ; perfect fourth   = P4
    #diatonic tritone = DT ; perfect fifth    = P5 ; minor sixth      = m6
    #major sixth      = M6 ; minor seventh    = m7 ; major seventh    = M7
    #octave           = O8
    harmony_select = {'U0' : 1,
                      'ST' : 16/15,
                      'M2' : 9/8,
                      'm3' : 6/5,
                      'M3' : 5/4,
                      'P4' : 4/3,
                      'DT' : 45/32,
                      'P5' : 3/2,
                      'm6': 8/5,
                      'M6': 5/3,
                      'm7': 9/5,
                      'M7': 15/8,
                      'O8': 2
                     }
    harmony = np.array([]) #This array will contain the song harmony
    harmony_val = harmony_select[harmonize] #This will select the ratio for the desired harmony
                                               
    song_freqs = np.array([]) #This array will contain the chosen frequencies used in our song :]
    song = np.array([])       #This array will contain the song signal
    octaves = np.array([0.5,1,2])#Go an octave below, same note, or go an octave above
    t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
    #Make a song with numpy array :]
    #nPixels = int(len(frequencies))#All pixels in image
    for k in range(nPixels):
        if useOctaves:
            octave = random.choice(octaves)
        else:
            octave = 1
        
        if randomPixels == False:
            val =  octave * frequencies[k]
        else:
            val = octave * random.choice(frequencies)
            
        #Make note and harmony note    
        note   = 0.5*np.sin(2*np.pi*val*t)
        h_note = 0.5*np.sin(2*np.pi*harmony_val*val*t)  
        
        #Place notes into corresponfing arrays
        song       = np.concatenate([song, note])
        harmony    = np.concatenate([harmony, h_note])                                     
        #song_freqs = np.concatenate([song_freqs, val])
                                               
    return song, pixels_df, harmony
def get_piano_notes():
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    base_freq = 440 #Frequency of Note A4
    keys = np.array([x+str(y) for y in range(0,9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end+1]
    
    note_freqs = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0 # stop
    return note_freqs

def get_sine_wave(frequency, duration, sample_rate=44100, amplitude=4096):
    t = np.linspace(0, duration, int(sample_rate*duration)) # Time axis
    wave = amplitude*np.sin(2*np.pi*frequency*t)
    return wave
def makeScale(whichOctave, whichKey, whichScale, makeHarmony = 'U0'):
    
    #Load note dictionary
    note_freqs = get_piano_notes()
    
    #Define tones. Upper case are white keys in piano. Lower case are black keys
    scale_intervals = ['A','a','B','C','c','D','d','E','F','f','G','g']
    
    #Find index of desired key
    index = scale_intervals.index(whichKey)
    
    #Redefine scale interval so that scale intervals begins with whichKey
    new_scale = scale_intervals[index:12] + scale_intervals[:index]
    
    #Choose scale
    if whichScale == 'AEOLIAN':
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'BLUES':
        scale = [0, 2, 3, 4, 5, 7, 9, 10, 11]
    elif whichScale == 'PHYRIGIAN':
        scale = [0, 1, 3, 5, 7, 8, 10]
    elif whichScale == 'CHROMATIC':
        scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    elif whichScale == 'DIATONIC_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'DORIAN':
        scale = [0, 2, 3, 5, 7, 9, 10]
    elif whichScale == 'HARMONIC_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 11]
    elif whichScale == 'LYDIAN':
        scale = [0, 2, 4, 6, 7, 9, 11]
    elif whichScale == 'MAJOR':
        scale = [0, 2, 4, 5, 7, 9, 11]
    elif whichScale == 'MELODIC_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 9, 10, 11]
    elif whichScale == 'MINOR':    
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'MIXOLYDIAN':     
        scale = [0, 2, 4, 5, 7, 9, 10]
    elif whichScale == 'NATURAL_MINOR':   
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'PENTATONIC':    
        scale = [0, 2, 4, 7, 9]
    else:
        print('Invalid scale name')
    
    #Make harmony dictionary (i.e. fundamental, perfect fifth, major third, octave)
    #unison           = U0
    #semitone         = ST
    #major second     = M2
    #minor third      = m3
    #major third      = M3
    #perfect fourth   = P4
    #diatonic tritone = DT
    #perfect fifth    = P5
    #minor sixth      = m6
    #major sixth      = M6
    #minor seventh    = m7
    #major seventh    = M7
    #octave           = O8
    harmony_select = {'U0' : 1,
                      'ST' : 16/15,
                      'M2' : 9/8,
                      'm3' : 6/5,
                      'M3' : 5/4,
                      'P4' : 4/3,
                      'DT' : 45/32,
                      'P5' : 3/2,
                      'm6': 8/5,
                      'M6': 5/3,
                      'm7': 9/5,
                      'M7': 15/8,
                      'O8': 2
                     }
    
    #Get length of scale (i.e., how many notes in scale)
    nNotes = len(scale)
    
    #Initialize arrays
    freqs = []
    #harmony = []
    #harmony_val = harmony_select[makeHarmony]
    for i in range(nNotes):
        note = new_scale[scale[i]] + str(whichOctave)
        freqToAdd = note_freqs[note]
        freqs.append(freqToAdd)
        #harmony.append(harmony_val*freqToAdd)
    return freqs#,harmony
test_scale,test_harmony = makeScale(3, 'a', 'HARMONIC_MINOR',makeHarmony = 'm6')
print(test_scale)
print(test_harmony)
#Pixel Art
pixel_art = cv2.imread('pixel_art1.png')
pixel_art2 = cv2.cvtColor(pixel_art, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(pixel_art2)
plt.grid(False)
plt.show()
pixel_scale = makeScale(3, 'a', 'HARMONIC_MINOR')
pixel_song, pixel_df,pixel_df_harmony = img2music(pixel_art, 
                                                  pixel_scale, 
                                                  T = 0.2, 
                                                  randomPixels = True)

wavfile.write('pixel_song2.wav'    , rate = 22050, data = pixel_song.astype(np.float32))
ipd.Audio(pixel_song, rate = sr)
print('pixel_df 3',pixel_df)
#Waterfall
waterfall = cv2.imread('waterfall.jpg')
waterfall2 = cv2.cvtColor(waterfall, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(waterfall2)
plt.grid(False)
plt.show()
waterfall_scale = makeScale(1, 'd', 'MAJOR')
waterfall_song, waterfall_df,waterfall_song_harmony  = img2music(waterfall, 
                                                                waterfall_scale, 
                                                                T = 0.3,
                                                                randomPixels = True, 
                                                                useOctaves = True)

wavfile.write('waterfall_song2.wav'    , rate = 22050, data = waterfall_song.astype(np.float32))
ipd.Audio(waterfall_song, rate = sr)
print('waterfall_df', waterfall_df)
#Peacock
peacock = cv2.imread('peacock.jpg')
peacock2 = cv2.cvtColor(peacock, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(peacock2)
plt.grid(False)
plt.show()
peacock_scale = makeScale(3, 'E', 'DORIAN')
peacock_song, peacock_df, peacock_song_harmony  = img2music(peacock, 
                                                            peacock_scale, 
                                                            T = 0.2, 
                                                            randomPixels = False,
                                                            useOctaves = True, nPixels = 120)

wavfile.write('peacock_song.wav'    , rate = 22050, data = peacock_song.astype(np.float32))
ipd.Audio(peacock_song, rate = sr)
print('peacock_df', peacock_df)
#Cat
cat = cv2.imread('cat1.jpg')
cat2 = cv2.cvtColor(cat, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(cat2)
plt.grid(False)
plt.show()
cat_scale = makeScale(2, 'f', 'AEOLIAN')
cat_song, cat_df,cat_song_harmony  = img2music(cat, 
                                               cat_scale, 
                                               T = 0.4, 
                                               randomPixels = True,
                                               useOctaves = True, 
                                               nPixels = 60)

wavfile.write('cat_song.wav'    , rate = 22050, data = cat_song.astype(np.float32))
ipd.Audio(cat_song, rate = sr)
print('cat_df', cat_df)
#water
water = cv2.imread('water.jpg')
water2 = cv2.cvtColor(water, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(water2)
plt.grid(False)
plt.show()
water_scale = makeScale(2, 'B', 'LYDIAN')
water_song, water_df, water_song_harmony  = img2music(water, 
                                                      water_scale, 
                                                      T = 0.2, 
                                                      randomPixels = False,
                                                      useOctaves = True, nPixels = 60)

wavfile.write('water_song2.wav'    , rate = 22050, data = water_song.astype(np.float32))
ipd.Audio(water_song, rate = sr)
print('water_df', water_df)
#earth
earth = cv2.imread('earth.jpg')
earth2 = cv2.cvtColor(earth, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(earth2)
plt.grid(False)
plt.show()
earth_scale = makeScale(3, 'g', 'MELODIC_MINOR')
earth_song, earth_df,earth_song_harmony  = img2music(earth, 
                                                     earth_scale, 
                                                     T = 0.3, 
                                                     randomPixels = False,
                                                     useOctaves = True, nPixels = 60)
wavfile.write('earth_song.wav'    , rate = 22050, data = earth_song.astype(np.float32))
ipd.Audio(earth_song, rate = sr)
print('earth_df', earth_df)
html = earth_df.to_html(max_rows = 10)
print(html)
#old_building
old_building = cv2.imread('old_building.jpeg')
old_building2 = cv2.cvtColor(old_building, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(old_building2)
plt.grid(False)
plt.show()
old_building_scale = makeScale(2, 'd', 'PHYRIGIAN')
old_building_song, old_building_df  = img2music(old_building, old_building_scale,
                                     T = 0.3, randomPixels = True, useOctaves = True, nPixels = 60)
ipd.Audio(old_building_song, rate = sr)
#mom
mom = cv2.imread('mami.jpg')
mom2 = cv2.cvtColor(mom, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(mom2)
plt.grid(False)
plt.show()
mom_scale = makeScale(3, 'g', 'MAJOR')
mom_song, mom_df  = img2music(mom, anto_scale,
                                     T = 0.3, randomPixels = True, useOctaves = True, nPixels = 60)
ipd.Audio(mom_song, rate = sr)
#%%
#old_building
catterina = cv2.imread('catterina.jpg')
catterina2 = cv2.cvtColor(catterina, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(catterina2)
plt.grid(False)
plt.show()
#%%
catterina_scale = makeScale(3, 'A', 'HARMONIC_MINOR')
catterina_song, catterina_df  = img2music(catterina, catterina_scale,
                                     T = 0.2, randomPixels = True, useOctaves = True, nPixels = 60)
ipd.Audio(catterina_song, rate = sr)
#%% md
# <p>Cool! The scale generator I made could easily accomodate new scales. Build your own scales :]</p>
# 
# <h2>Exporting song into a .wav file</h2>
# <p>The following code can be used to export the song into a .wav file. Since the numpy arrays we are generating are dtype = float32 we need to specifiy that in the data paramter.</p>
#%%
from scipy.io import wavfile
wavfile.write('earth_song.wav'    , rate = 22050, data = earth_song.astype(np.float32))
wavfile.write('water_song.wav'    , rate = 22050, data = water_song.astype(np.float32))
wavfile.write('catterina_song.wav', rate = 22050, data = catterina_song.astype(np.float32))
#%% md
# I'll also do it now for an example in which I'm using harmony
#%%
#nature
nature = cv2.imread('nature1.webp')
nature2 = cv2.cvtColor(nature, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(nature2)
plt.grid(False)
plt.show()
#%%
nature_scale = makeScale(3, 'a', 'HARMONIC_MINOR')
nature_song, nature_df, nature_harmony = img2music(nature, nature_scale, 
                                                 T = 0.2, randomPixels = True, harmonize = 'm3')

wavfile.write('nature_song1.wav', rate = 22050, data = nature_song.astype(np.float32))

#This is the original song we made from the picture
ipd.Audio(nature_song, rate = sr)
#%%
#This is the harmony to the song we made from the picture
wavfile.write('nature_song2.wav', rate = 22050, data = nature_harmony.astype(np.float32))
ipd.Audio(nature_harmony, rate = sr)
#%% md
# <p>The song and harmony arrays are both 1D. I can combine them into a 2D array using <code>np.vstack</code>. This will allow us to save our harmonized song into a single .wav file :]</p>
#%%
nature_harmony_combined = np.vstack((nature_song, nature_harmony))
wavfile.write('nature_harmony_combined.wav',
              rate = 22050,
              data = nature_harmony_combined.T.astype(np.float32))
ipd.Audio(nature_harmony_combined, rate = sr)
#%%
print(nature_harmony_combined.shape)
#%% md
# <p>From the documentation for scipy.io.wavfile.write, if want to write a 2D array into a .wav file, the 2D array must be have dimensions in the form of (Nsamples, Nchannels). Notice how the shape of our array is currently (2, 264600). This means we have Nchannels = 2 and Nsamples = 264600. To ensure our numpy array has the correct shape for scipy.io.wavfile.write I'll transpose the array first.</p>
#%%
wavfile.write('nature_harmony_combined.wav', rate = 22050, 
              data = nature_harmony_combined.T.astype(np.float32))
#%% md
# <h2>Adding Effects to Our Music with Pedalboard</h2>
# 
# Now I'm going to load the .wav files and do some extra  manipulation on it using the pedalboard module from Spotify. You can read more about the pedalboard library <a href = "https://github.com/spotify/pedalboard">here</a> and <a href = "https://spotify.github.io/pedalboard/reference/pedalboard.html">here</a>.
#%%
from pedalboard import Pedalboard, Chorus, Reverb, Compressor, Gain, LadderFilter 
from pedalboard import Phaser, Delay, PitchShift, Distortion
from pedalboard.io import AudioFile
# Read in a whole audio file:
with AudioFile('water_song.wav', 'r') as f:
    audio = f.read(f.frames)
    samplerate = f.samplerate

# Make a Pedalboard object, containing multiple plugins:
board = Pedalboard([
    #Delay(delay_seconds=0.25, mix=1.0),
    Compressor(threshold_db=-100, ratio=25),
    Gain(gain_db=150),
    Chorus(),
    LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=900),
    Phaser(),
    Reverb(room_size=0.5),
])

# Run the audio through this pedalboard!
effected = board(audio, samplerate)

# Write the audio back as a wav file:
with AudioFile('processed-water_song.wav', 'w', samplerate, effected.shape[0]) as f:
    f.write(effected)

ipd.Audio('processed-water_song.wav')
#%%
# Read in a whole audio file:
with AudioFile('catterina_song.wav', 'r') as f:
    audio = f.read(f.frames)
    samplerate = f.samplerate
print(samplerate)
# Make a Pedalboard object, containing multiple plugins:
board = Pedalboard([
    LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=100),
    Delay(delay_seconds = 0.3),
    Reverb(room_size = 0.6, wet_level=0.2, width = 1.0),
    PitchShift(semitones = 6),
])

# Run the audio through this pedalboard!
effected = board(audio, samplerate)

# Write the audio back as a wav file:
with AudioFile('processed-catterina_song.wav', 'w', samplerate, effected.shape[0]) as f:
    f.write(effected)

ipd.Audio('processed-catterina_song.wav')
#%%
# Read in a whole audio file:
with AudioFile('nature_harmony_combined.wav', 'r') as f:
    audio = f.read(f.frames)
    samplerate = f.samplerate
# Make a Pedalboard object, containing multiple plugins:
board = Pedalboard([
    LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=100),
    Delay(delay_seconds = 0.1),
    Reverb(room_size = 1, wet_level=0.1, width = 0.5),
    PitchShift(semitones = 6),
    #Chorus(rate_hz = 15),
    Phaser(rate_hz = 5, depth = 0.5, centre_frequency_hz = 500.0),
])

# Run the audio through this pedalboard!
effected = board(audio, samplerate)

# Write the audio back as a wav file:
with AudioFile('processed-nature_harmony_combined.wav', 'w', samplerate, effected.shape[0]) as f:
    f.write(effected)

ipd.Audio('processed-nature_harmony_combined.wav')
#%% md
# Neat!
#%% md
# <h2>Using Librosa For Mapping Other Musical Quantities</h2>
# 
# <p>Librosa is a wonderful package that allows one to carry out a variety of operations on sound data. Here I used it to readily convert frequencies into 'Notes' and 'Midi Numbers'.</p>
#%%
#Convert frequency to a note
catterina_df['notes'] = catterina_df.apply(lambda row : librosa.hz_to_note(row['frequencies']), 
                                           axis = 1)  
#Convert note to a midi number
catterina_df['midi_number'] = catterina_df.apply(lambda row : librosa.note_to_midi(row['notes']), 
                                                 axis = 1)    
catterina_df
#%%
html = catterina_df.to_html()
print(html)
#%% md
# <h2>Making a MIDI from our Song</h2>
# 
# <p>Now that I've generated a dataframe containing frequencies, notes and midi numbers I can make a midi file out of it! I could then use this MIDI file to generate sheet music for our song :]</p>
# 
# <p>To make a MIDI file, I'll make use of the <code>midiutil</code> package. This package allows us to build MIDI files from an array of MIDI numbers. You can configure your file in a variety of ways by setting up volume, tempos and tracks. For now, I'll just make a single track midi file</p>
#%%
#Convert midi number column to a numpy array
midi_number = catterina_df['midi_number'].to_numpy()
#%%
degrees  = list(midi_number) # MIDI note number
track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 240  # In BPM
volume   = 100 # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1) # One track, defaults to format 1 (tempo track
                     # automatically created)
MyMIDI.addTempo(track,time, tempo)

for pitch in degrees:
    MyMIDI.addNote(track, channel, pitch, time, duration, volume)
    time = time + 1
with open("catterina.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)
#%% md
# <h2>Converting hues to frequencies (2nd Idea)</h2>
# 
# <p>My second idea to convert color into sound was via a 'spectral' method. This is something that I'm still playing around with.</p>
#%%
#Convert hue to wavelength[nm] via interpolation. Assume spectrum is contained between 400-650nm
def hue2wl(h, wlMax = 650, wlMin = 400, hMax = 270, hMin = 0):
    #h *= 2
    hMax /= 2
    hMin /= 2
    wlRange = wlMax - wlMin
    hRange = hMax - hMin
    wl =  wlMax - ((h* (wlRange))/(hRange))
    return wl
#%%
#Array with hue values from 0 degrees to 270 degrees
h_array = np.arange(0,270,1)
h_array.shape

# define vectorized sigmoid
hue2wl_v = np.vectorize(hue2wl)
test = hue2wl_v(h_array)
test.shape
np.min(test)

plt.title("Interpolation of Hue and Wavelength") 
plt.xlabel("Hue()") 
plt.ylabel("Wavelength[nm]") 
plt.scatter(h_array, test, c = cm.gist_rainbow_r(np.abs(h_array)), edgecolor='none')
plt.gca().invert_yaxis()
plt.style.use('seaborn-darkgrid')
plt.show()
#%%
img = cv2.imread('colors.jpg')
#Convert a hue value to wavelength via interpolation
#Assume that visible spectrum is contained between 400-650nm
def hue2wl(h, wlMax = 650, wlMin = 400, hMax = 270, hMin = 0):
    #h *= 2
    hMax /= 2
    hMin /= 2
    wlRange = wlMax - wlMin
    hRange = hMax - hMin
    wl =  wlMax - ((h* (wlRange))/(hRange))
    return wl

def wl2freq(wl):
    wavelength = wl
    sol = 299792458.00 #this is the speed of light in m/s
    sol *= 1e9 #Convert speed of light to nm/s
    freq = (sol / wavelength) * (1e-12)
    return freq

def img2music2(img, fName):
    
    #Get height and width of image
    height, width, _ = img.shape
    
    #Convet from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #Populate hues array with H channel for each pixel
    i=0 ; j=0
    hues = []
    for i in range(height):
        for j in range(width):
            hue = hsv[i][j][0] #This is the hue value at pixel coordinate (i,j)
            hues.append(hue)
            
    #Make pandas dataframe        
    hues_df = pd.DataFrame(hues, columns=['hues'])
    hues_df['nm'] = hues_df.apply(lambda row : hue2wl(row['hues']), axis = 1)  
    hues_df['freq'] = hues_df.apply(lambda row : wl2freq(row['nm']), axis = 1) 
    hues_df['notes'] = hues_df.apply(lambda row : librosa.hz_to_note(row['freq']), axis = 1)  
    hues_df['midi_number'] = hues_df.apply(lambda row : librosa.note_to_midi(row['notes']), axis = 1) 
    
    print("Done making song from image!") 
    
    return hues_df
#%%
df = img2music2(img,'color')
df
#%%
html = df.to_html()
print(html)
#%%
#Convert midi number column to a numpy array
sr = 22050 # sample rate
song = df['freq'].to_numpy()
ipd.Audio(song, rate = sr) # load a NumPy array
#%%
a_HarmonicMinor = [220.00, 246.94 ,261.63, 293.66, 329.63, 349.23, 415.30, 440.00] 
frequencies = df['freq'].to_numpy()
song = np.array([]) 
harmony = np.array([]) 
octaves = np.array([1/4,1,2,1,2])
sr = 22050 # sample rate
T = 0.25    # 0.1 second duration
t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
#Make a song with numpy array :]
nPixels = int(len(frequencies)/height)
nPixels = 30
#for j in tqdm(range(nPixels), desc="Processing Frame"):#Add progress bar for frames processed      
for i in range(nPixels):  
    octave = random.choice(octaves)
    val =  octave * frequencies[i]
    note  = 0.5*np.sin(2*np.pi*val*t)
    song  = np.concatenate([song, note])
ipd.Audio(song, rate=sr) # load a NumPy array
#%% md
# <h2>Conclusion</h2>
# I showed how musis can be made from images and how our songs can be exported into .wav files for subsequent processing. There's tons of expereimentation that can be done with this. I had fun making this project and I hope you have using it and building upon it!