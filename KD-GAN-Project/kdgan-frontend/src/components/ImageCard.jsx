import React from 'react';

function ImageCard({ image, caption }) {
  return (
    <div className="image-container">
      <img src={image || "/placeholder.svg"} alt={caption} />
      {caption && <div className="image-caption">{caption}</div>}
    </div>
  );
}

export default ImageCard;