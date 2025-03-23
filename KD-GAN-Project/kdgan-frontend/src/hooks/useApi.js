import { useState, useCallback } from 'react';
import axios from 'axios';

// Create axios instance
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generateImage = useCallback(async (text) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/generate', { text });
      setLoading(false);
      return response.data;
    } catch (err) {
      setLoading(false);
      setError(err.response?.data?.message || 'An error occurred while generating the image');
      return null;
    }
  }, []);

  const getGalleryImages = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.get('/gallery');
      setLoading(false);
      return response.data;
    } catch (err) {
      setLoading(false);
      setError(err.response?.data?.message || 'An error occurred while fetching gallery images');
      return [];
    }
  }, []);

  return {
    loading,
    error,
    generateImage,
    getGalleryImages,
  };
}