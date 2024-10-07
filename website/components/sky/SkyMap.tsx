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
      map.fitBounds(imageBounds);
    }, [map, imageBounds]);
    return null;
  };

  return (
    <MapContainer
      center={[0, 0]}
      zoom={0}
      style={{ height: '100vh', width: '100%' }}
      crs={L.CRS.Simple}
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