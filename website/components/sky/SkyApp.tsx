"use client";

import React, { useState, useRef } from 'react';
import dynamic from 'next/dynamic';
import L from 'leaflet';
import Image from 'next/image';

const SkyMap = dynamic(() => import('./SkyMap'), {
  ssr: false,
  loading: () => <p>Loading map...</p>
});

const sampleFits = [
  { id: 1, name: "Sample FITS 1", url: "/path/to/sample1.fits" },
  { id: 2, name: "Sample FITS 2", url: "/path/to/sample2.fits" },
  { id: 3, name: "Sample FITS 3", url: "/path/to/sample3.fits" },
  { id: 4, name: "Sample FITS 4", url: "/path/to/sample4.fits" },
  { id: 5, name: "Sample FITS 5", url: "/path/to/sample5.fits" },
];

const sampleImages = [
  { id: 1, name: "Sample Image 1", url: "/jwst/1.jpg" },
  { id: 2, name: "Sample Image 2", url: "/jwst/2.jpg" },
  { id: 3, name: "Sample Image 3", url: "/jwst/3.jpg" },
  { id: 4, name: "Sample Image 4", url: "/jwst/4.jpg" },
  { id: 5, name: "Sample Image 5", url: "/jwst/5.png" },
];

const SkyApp: React.FC = () => {
  const [option, setOption] = useState<string>('');
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [musicUrl, setMusicUrl] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleRegionSelect = async (bounds: L.LatLngBounds) => {
    try {
      const response = await fetch('/api/convert-to-music', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          southWest: bounds.getSouthWest(),
          northEast: bounds.getNorthEast(),
        }),
      });

      if (!response.ok) throw new Error('Failed to convert image to music');
      const musicUrl = await response.text();
      setMusicUrl(musicUrl);
    } catch (error) {
      console.error('Error converting image to music:', error);
      setErrorMessage('Failed to convert region to music. Please try again.');
    }
  };

  const handleSampleSelect = async (url: string) => {
    setSelectedSample(url);
    setMusicUrl('/path/to/sample-music.wav');
    setErrorMessage('');
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (option === 'fitsFile' && !file.name.toLowerCase().endsWith('.fits')) {
      setErrorMessage('Please upload only .fits files');
      setUploadedFile(null);
      event.target.value = '';
      return;
    }

    setUploadedFile(file);
    setErrorMessage('');
    setMusicUrl('/path/to/uploaded-music.wav');
  };

  return (
    <section className="min-h-screen w-full flex flex-col items-center justify-start pt-20 pb-28 px-4">
      <div className="max-w-7xl w-full space-y-12">
        <div className="text-center space-y-4">
          <h1 className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-500 via-blue-500 to-cyan-500">
            Space Sounds Explorer
          </h1>
          <p className="text-xl text-gray-300">
            Transform celestial wonders into musical experiences
          </p>
        </div>

        <div className="flex flex-wrap justify-center gap-6">
          {['skyView', 'fitsFile', 'spaceImage'].map((opt) => (
            <button
              key={opt}
              onClick={() => {
                setOption(opt);
                setErrorMessage('');
                setUploadedFile(null);
              }}
              className={`px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300 transform hover:scale-105 ${
                option === opt
                  ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              {opt === 'skyView' ? 'JWST Sky View' : 
               opt === 'fitsFile' ? 'JWST Fits File' : 'JWST Space Image'}
            </button>
          ))}
        </div>

        {option && (
          <div className="bg-gray-900/60 backdrop-blur-md rounded-2xl p-8 shadow-xl">
            {option === 'skyView' && (
              <div className="h-[600px] w-full rounded-xl overflow-hidden">
                <SkyMap
                  imageUrl="/jwst.webp"
                  onRegionSelect={handleRegionSelect}
                />
              </div>
            )}

            {(option === 'fitsFile' || option === 'spaceImage') && (
              <div className="space-y-8">
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
                  {(option === 'fitsFile' ? sampleFits : sampleImages).map((sample) => (
                    option === 'spaceImage' ? (
                      <div
                        key={sample.id}
                        className="aspect-square relative rounded-lg overflow-hidden cursor-pointer group"
                        onClick={() => handleSampleSelect(sample.url)}
                      >
                        <Image
                          src={sample.url}
                          alt={sample.name}
                          fill
                          className="object-cover transition-transform duration-300 group-hover:scale-110"
                        />
                      </div>
                    ) : (
                      <button
                        key={sample.id}
                        onClick={() => handleSampleSelect(sample.url)}
                        className="p-4 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors duration-300"
                      >
                        {sample.name}
                      </button>
                    )
                  ))}
                </div>

                <div className="flex flex-col items-center gap-4">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 rounded-xl text-white font-semibold hover:opacity-90 transition-opacity"
                  >
                    Upload {option === 'fitsFile' ? 'FITS File' : 'Space Image'}
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept={option === 'fitsFile' ? '.fits' : 'image/*'}
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  {errorMessage && (
                    <p className="text-red-500 font-medium">{errorMessage}</p>
                  )}
                  {uploadedFile && (
                    <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                      <p className="text-green-400">
                        Uploaded: {uploadedFile.name}
                      </p>
                      {option === 'spaceImage' && (
                        <div className="mt-4 max-w-md mx-auto">
                          <Image
                            src={URL.createObjectURL(uploadedFile)}
                            alt="Uploaded image"
                            width={400}
                            height={400}
                            className="rounded-lg"
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}

            {musicUrl && (
              <div className="mt-8 space-y-4">
                <audio
                  controls
                  src={musicUrl}
                  className="w-full h-12 rounded-lg"
                >
                  Your browser does not support the audio element.
                </audio>
                <div className="flex justify-center">
                  <a
                    href={musicUrl}
                    download="space-music.wav"
                    className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl text-white font-semibold hover:opacity-90 transition-opacity"
                  >
                    Download WAV
                  </a>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

     
    </section>
  );
};

export default SkyApp;