import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import GeneratePage from './pages/GeneratePage';
import GalleryPage from './pages/GalleryPage';
import AboutPage from './pages/AboutPage';

function App() {
  return (
    <div className="app">
      <Header />
      <main className="main">
        <div className="container">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/generate" element={<GeneratePage />} />
            <Route path="/gallery" element={<GalleryPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </div>
      </main>
      <Footer />
    </div>
  );
}

export default App;