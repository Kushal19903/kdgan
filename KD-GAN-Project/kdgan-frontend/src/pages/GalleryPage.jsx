import React, { useState, useEffect } from 'react';
import ImageCard from '../components/ImageCard';
import Loading from '../components/Loading';
import Alert from '../components/Alert';
import { useApi } from '../hooks/useApi';

function GalleryPage() {
  const [images, setImages] = useState([]);
  const { loading, error, getGalleryImages } = useApi();

  useEffect(() => {
    const fetchImages = async () => {
      const data = await getGalleryImages();
      if (data) {
        setImages(data);
      }
    };
    
    fetchImages();
  }, [getGalleryImages]);

  // For demo purposes, if no API is available
  const demoImages = [
    {
      id: 1,
      url: 'https://via.placeholder.com/400x400?text=Generated+Image+1',
      caption: 'A beautiful sunset over the mountains with a lake in the foreground'
    },
    {
      id: 2,
      url: 'https://via.placeholder.com/400x400?text=Generated+Image+2',
      caption: 'A futuristic city with flying cars and tall skyscrapers'
    },
    {
      id: 3,
      url: 'https://via.placeholder.com/400x400?text=Generated+Image+3',
      caption: 'A serene forest with sunlight filtering through the trees'
    },
    {
      id: 4,
      url: 'https://via.placeholder.com/400x400?text=Generated+Image+4',
      caption: 'A cozy cabin in the woods during winter with smoke coming from the chimney'
    },
    {
      id: 5,
      url: 'https://via.placeholder.com/400x400?text=Generated+Image+5',
      caption: 'An underwater scene with colorful coral reefs and tropical fish'
    },
    {
      id: 6,
      url: 'https://via.placeholder.com/400x400?text=Generated+Image+6',
      caption: 'A medieval castle on a hill with a village below'
    }
  ];

  return (
    <div>
      <h1 className="text-center mb-4">Image Gallery</h1>
      
      {error && <Alert type="danger" message={error} />}
      
      {loading ? (
        <Loading message="Loading gallery images..." />
      ) : (
        <>
          {images.length > 0 ? (
            <div className="grid">
              {images.map((image) => (
                <ImageCard 
                  key={image.id} 
                  image={image.url} 
                  caption={image.caption} 
                />
              ))}
            </div>
          ) : (
            <div className="grid">
              {demoImages.map((image) => (
                <ImageCard 
                  key={image.id} 
                  image={image.url} 
                  caption={image.caption} 
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default GalleryPage;