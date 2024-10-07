from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

file_path = 'sample.fits'

with fits.open(file_path) as hdul:
    hdul.info() 
    
    sci_data = hdul['SCI'].data

if sci_data is not None:
    sci_data = np.clip(sci_data, np.percentile(sci_data, 1), np.percentile(sci_data, 99))

    plt.figure(figsize=(8, 8))
    plt.imshow(np.log1p(sci_data),  origin='lower')
    plt.colorbar()
    plt.title("JWST Science Image (SCI) - Log Scale")
    plt.show()
else:
    print("No image data found in the SCI extension.")