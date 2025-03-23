import React from 'react';
import { Link } from 'react-router-dom';
import Button from '../components/Button';

function HomePage() {
  return (
    <div>
      <section className="card mb-4">
        <h1 className="text-center mb-3">Knowledge Driven GAN for Text to Image Synthesis</h1>
        <p className="mb-3">
          Welcome to KD-GAN, a state-of-the-art text-to-image synthesis system that leverages the power of BERT language model
          and Generative Adversarial Networks to create high-quality images from textual descriptions.
        </p>
        <div className="flex justify-center mt-3">
          <Link to="/generate">
            <Button variant="primary">Try It Now</Button>
          </Link>
        </div>
      </section>
      
      <section className="grid">
        <div className="card">
          <h2 className="mb-2">BERT Integration</h2>
          <p>
            Our model uses BERT (Bidirectional Encoder Representations from Transformers) to understand the semantic meaning
            of your text descriptions, enabling more accurate and contextually relevant image generation.
          </p>
        </div>
        
        <div className="card">
          <h2 className="mb-2">Knowledge Driven</h2>
          <p>
            Unlike traditional GANs, our approach incorporates knowledge from pre-trained language models to enhance
            the quality and semantic relevance of generated images.
          </p>
        </div>
        
        <div className="card">
          <h2 className="mb-2">High Quality Results</h2>
          <p>
            The KD-GAN architecture produces high-resolution, detailed images that accurately reflect the input text
            descriptions, with improved visual quality compared to standard text-to-image models.
          </p>
        </div>
      </section>
    </div>
  );
}

export default HomePage;