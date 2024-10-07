"use client";

import React, { useState, useRef, useEffect } from 'react';
import dynamic from 'next/dynamic';
import L from 'leaflet';
import Image from 'next/image';

const sampleImages = [
  { id: 1, name: "Sample Image 1", url: "/jwst/1.jpg", sound: "/images/1.wav" },
  { id: 2, name: "Sample Image 2", url: "/jwst/2.jpg", sound: "/images/2.wav" },
  { id: 3, name: "Sample Image 3", url: "/jwst/3.jpg", sound: "/images/3.wav" },
  { id: 4, name: "Sample Image 4", url: "/jwst/4.jpg", sound: "/images/4.wav" },
  { id: 5, name: "Sample Image 5", url: "/jwst/5.png", sound: "/images/5.wav" },
];

const randomSounds = [
  "/random/1.wav",
  "/random/2.wav",
  "/random/3.wav",
  "/random/4.wav",
"/random/5.wav",
"/random/6.wav",
"/random/7.wav",
];

const sampleFitsFiles = [
  { id: 1, name: "Sample DATA FITS 1", sound: "/fits/1.wav" },
  { id: 2, name: "Sample DATA FITS 2", sound: "/fits/2.wav" },
  { id: 3, name: "Sample DATA FITS 3", sound: "/fits/3.wav" },
];

const SkyApp: React.FC = () => {
  const [option, setOption] = useState<string>('');
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [musicUrl, setMusicUrl] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null); // Reference to control audio

  // State for audio playback
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const handleRegionSelect = (bounds: L.LatLngBounds) => {
    const randomSound = randomSounds[Math.floor(Math.random() * randomSounds.length)];
    playNewAudio(randomSound);
  };

  const handleSampleSelect = (url: string, sound: string) => {
    setSelectedSample(url);
    playNewAudio(sound);  // Ensure the selected sound is played
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
    
    // Assign a random sound from the list for both image and FITS file uploads
    const randomSound = randomSounds[Math.floor(Math.random() * randomSounds.length)];
    playNewAudio(randomSound); // Play a sound when a new file is uploaded
  };

  const playNewAudio = (audioUrl: string) => {
    // Stop any current audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    // Set and play the new audio
    setMusicUrl(audioUrl);
    setIsPlaying(true); // Set to playing when audio is loaded
  };

  useEffect(() => {
    // Play the new audio if there's an updated musicUrl
    if (musicUrl && audioRef.current) {
      audioRef.current.play();
    }
  }, [musicUrl]);

  useEffect(() => {
    const audioElement = audioRef.current;

    if (audioElement) {
      const handleTimeUpdate = () => {
        setCurrentTime(audioElement.currentTime);
        setDuration(audioElement.duration);
      };

      audioElement.addEventListener('timeupdate', handleTimeUpdate);

      return () => {
        audioElement.removeEventListener('timeupdate', handleTimeUpdate);
      };
    }
  }, [musicUrl]);

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (audioRef.current) {
      audioRef.current.currentTime = Number(event.target.value);
    }
  };

  const downloadAudio = () => {
    if (musicUrl) {
      const link = document.createElement('a');
      link.href = musicUrl;
      link.download = musicUrl.split('/').pop() || 'audio.wav';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <section className="min-h-screen w-full flex flex-col items-center justify-start pt-20 pb-28 px-4"   id="sky">
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
          {[ 'fitsFile', 'spaceImage'].map((opt) => (
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
               opt === 'fitsFile' ? 'JWST FITS File' : 'JWST Space Image'}
            </button>
          ))}
        </div>

        {option && (
          <div className="bg-gray-900/60 backdrop-blur-md rounded-2xl p-8 shadow-xl">
            {option === 'fitsFile' && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {sampleFitsFiles.map((fits) => (
                    <div
                      key={fits.id}
                      className="aspect-square relative rounded-lg overflow-hidden cursor-pointer group"
                      onClick={() => handleSampleSelect("", fits.sound)}
                    >
                      <div className="flex justify-center items-center h-full bg-gray-800 text-white">
                        {fits.name}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex flex-col items-center gap-4">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 rounded-xl text-white font-semibold hover:opacity-90 transition-opacity"
                  >
                    Upload FITS File
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
                      <p className="text-white">{uploadedFile.name}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {option === 'spaceImage' && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
                  {sampleImages.map((sample) => (
                    <div
                      key={sample.id}
                      className="aspect-square relative rounded-lg overflow-hidden cursor-pointer group"
                      onClick={() => handleSampleSelect(sample.url, sample.sound)}
                    >
                      <Image
                        src={sample.url}
                        alt={sample.name}
                        layout="fill"
                        className="object-cover transition-transform duration-300 transform group-hover:scale-110"
                      />
                      <div className="absolute inset-0 flex justify-center items-center bg-black/30 text-white opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                        {sample.name}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex flex-col items-center mt-8">
              <audio
                ref={audioRef}
                src={musicUrl || ""}
                onEnded={() => setIsPlaying(false)}
                preload="metadata"
              />
              <div className="flex items-center space-x-4">
                <button
                  onClick={handlePlayPause}
                  className={`px-4 py-2 rounded-md text-white ${isPlaying ? 'bg-red-500' : 'bg-green-500'}`}
                >
                  {isPlaying ? 'Pause' : 'Play'}
                </button>
                <div className="flex items-center">
                  <input
                    type="range"
                    min={0}
                    max={duration}
                    value={currentTime}
                    onChange={handleSliderChange}
                    className="w-32"
                  />
                  <span className="text-white">{Math.floor(currentTime)} / {Math.floor(duration)}</span>
                </div>
              </div>
              {musicUrl && (
                <button
                  onClick={downloadAudio}
                  className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-md"
                >
                  Download Audio
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default SkyApp;
