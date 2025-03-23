import React from 'react';

function Alert({ type = 'success', message, onClose }) {
  if (!message) return null;
  
  return (
    <div className={`alert alert-${type}`}>
      {message}
      {onClose && (
        <button 
          type="button" 
          className="close" 
          onClick={onClose}
          style={{ float: 'right', background: 'none', border: 'none', cursor: 'pointer' }}
        >
          &times;
        </button>
      )}
    </div>
  );
}

export default Alert;