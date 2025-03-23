import React from 'react';

function Loading({ message = 'Loading...' }) {
  return (
    <div className="loading">
      <div className="flex flex-col items-center">
        <div className="spinner"></div>
        <p className="mt-2">{message}</p>
      </div>
    </div>
  );
}

export default Loading;