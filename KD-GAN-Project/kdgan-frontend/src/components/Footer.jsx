import React from 'react';

function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <p>&copy; {new Date().getFullYear()} KD-GAN: Knowledge Driven GAN for Text to Image Synthesis</p>
      </div>
    </footer>
  );
}

export default Footer;