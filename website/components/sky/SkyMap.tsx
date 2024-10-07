"use client";

import React, { useState, useEffect } from 'react';
import { MapContainer, ImageOverlay, useMapEvents, Rectangle, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

interface SkyMapProps {
  imageUrl: string;
  onRegionSelect: (bounds: L.LatLngBounds) => void;
}

const SkyMap: React.FC<SkyMapProps> = ({ imageUrl, onRegionSelect }) => {
  const [selection, setSelection] = useState<L.LatLngBounds | null>(null);
  const [imageBounds, setImageBounds] = useState<L.LatLngBounds>(L.latLngBounds([0, 0], [1, 1]));

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const { width, height } = img;
      setImageBounds(L.latLngBounds([0, 0], [height, width]));
    };
    img.src = imageUrl;
  }, [imageUrl]);

  const SelectionHandler = () => {
    const map = useMapEvents({
      mousedown: (e) => {
        setSelection(L.latLngBounds(e.latlng, e.latlng));
      },
      mousemove: (e) => {
        if (selection) {
          setSelection(L.latLngBounds(selection.getSouthWest(), e.latlng));
        }
      },
      mouseup: () => {
        if (selection) {
          onRegionSelect(selection);
          setSelection(null);
        }
      },
    });
    return null;
  };

  const MapAdjuster = () => {
    const map = useMap();
    
    useEffect(() => {
      // Initial map setup
      map.fitBounds(imageBounds);
      
      // Disable drag and set zoom constraints
      map.dragging.disable();
      map.setMinZoom(1);
      map.setMaxZoom(4);
      
      // Prevent the map from being moved outside the image bounds
      map.setMaxBounds(imageBounds.pad(0.1)); // Add 10% padding to prevent white edges
    }, [map, imageBounds]);

    return null;
  };

  return (
    <MapContainer
      center={[0, 0]}
      zoom={1}
      style={{ height: '100vh', width: '100%' }}
      crs={L.CRS.Simple}
      zoomControl={true}
      dragging={false}
      minZoom={1}
      maxZoom={4}
      maxBounds={imageBounds}
      maxBoundsViscosity={1.0}
    >
      <ImageOverlay
        url={imageUrl}
        bounds={imageBounds}
      />
      <SelectionHandler />
      <MapAdjuster />
      {selection && (
        <Rectangle bounds={selection} pathOptions={{ color: 'red' }} />
      )}
    </MapContainer>
  );
};

export default SkyMap;