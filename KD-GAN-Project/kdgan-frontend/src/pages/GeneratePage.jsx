import React, { useState } from 'react';
import TextInput from '../components/TextInput';
import Button from '../components/Button';
import Loading from '../components/Loading';
import Alert from '../components/Alert';
import { useApi } from '../hooks/useApi';

function GeneratePage() {
  const [text, setText] = useState('');
  const [generatedImage, setGeneratedImage] = useState(null);
  const { loading, error, generateImage } = useApi();

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      return;
    }
    
    const result = await generateImage(text);
    if (result) {
      setGeneratedImage(result.imageUrl);
    }
  };

  return (
    <div>
      <h1 className="text-center mb-4">Generate Images from Text</h1>
      
      {error && <Alert type="danger" message={error} />}
      
      <div className="card mb-4">
        <form onSubmit={handleSubmit}>
          <TextInput
            label="Enter a detailed description of the image you want to generate"
            value={text}
            onChange={handleTextChange}
            placeholder="A beautiful sunset over the mountains with a lake in the foreground..."
            isTextarea={true}
            required={true}
            id="text-input"
          />
          
          <div className="flex justify-center mt-3">
            <Button 
              type="submit" 
              variant="primary" 
              disabled={loading || !text.trim()}
            >
              {loading ? 'Generating...' : 'Generate Image'}
            </Button>
          </div>
        </form>
      </div>
      
      {loading ? (
        <Loading message="Generating your image. This may take a moment..." />
      ) : generatedImage && (
        <div className="card">
          <h2 className="mb-2">Generated Image</h2>
          <div className="image-container mb-3">
            <img src={generatedImage || "/placeholder.svg"} alt={text} />
          </div>
          <div className="mb-2">
            <strong>Description:</strong> {text}
          </div>
          <div className="flex justify-center">
            <a href={generatedImage} download="generated-image.png">
              <Button variant="secondary">Download Image</Button>
            </a>
          </div>
        </div>
      )}
      
      <div className="card mt-4">
        <h3 className="mb-2">Tips for Better Results</h3>
        <ul className="mb-0" style={{ paddingLeft: '1.5rem' }}>
          <li>Be specific and detailed in your descriptions</li>
          <li>Include information about colors, lighting, and composition</li>
          <li>Specify the style you want (e.g., photorealistic, cartoon, painting)</li>
          <li>Mention the main subject and background elements</li>
        </ul>
      </div>
    </div>
  );
}

export default GeneratePage;