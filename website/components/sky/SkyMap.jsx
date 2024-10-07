import React from 'react';
import { GlassMagnifier } from 'react-image-magnifiers';
import ImageZoom from 'react-image-zoom';

const ImageViewer = () => {
  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      
      <ImageZoom 
      image={{
        src: '/jwst.webp',
        alt: 'Image description',
        className: 'your-image-class'
      }}
      zoomPosition="original"
    />
    </div>
  );
};


export default MyComponent;