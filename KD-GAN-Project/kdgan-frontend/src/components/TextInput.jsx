import React from 'react';

function TextInput({ 
  label, 
  value, 
  onChange, 
  placeholder = '', 
  isTextarea = false,
  required = false,
  name = '',
  id = '',
  className = ''
}) {
  return (
    <div className={`form-group ${className}`}>
      {label && (
        <label htmlFor={id || name} className="form-label">
          {label} {required && <span className="text-danger">*</span>}
        </label>
      )}
      
      {isTextarea ? (
        <textarea
          id={id || name}
          name={name}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          required={required}
          className="form-control textarea"
        />
      ) : (
        <input
          type="text"
          id={id || name}
          name={name}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          required={required}
          className="form-control"
        />
      )}
    </div>
  );
}

export default TextInput;