import React from 'react';

function AboutPage() {
  return (
    <div>
      <h1 className="text-center mb-4">About KD-GAN</h1>
      
      <div className="card mb-4">
        <h2 className="mb-2">Project Overview</h2>
        <p>
          Knowledge Driven Generative Adversarial Network (KD-GAN) is a state-of-the-art text-to-image synthesis system
          that combines the power of BERT's language understanding with GANs to generate high-quality images from textual descriptions.
        </p>
        <p className="mb-0">
          This project aims to improve the quality and semantic relevance of generated images by leveraging the contextual
          understanding of language provided by BERT, a powerful pre-trained language model.
        </p>
      </div>
      
      <div className="card mb-4">
        <h2 className="mb-2">Technical Architecture</h2>
        <h3 className="mb-1">BERT Text Encoder</h3>
        <p className="mb-2">
          We use BERT (Bidirectional Encoder Representations from Transformers) to extract rich semantic features from
          the input text descriptions. These features capture the contextual meaning of words and phrases, enabling more
          accurate image generation.
        </p>
        
        <h3 className="mb-1">Generator</h3>
        <p className="mb-2">
          Our generator uses a deep convolutional neural network with residual blocks to transform random noise and text
          embeddings into high-quality images. The architecture includes multiple upsampling layers to progressively
          increase the resolution of the generated images.
        </p>
        
        <h3 className="mb-1">Discriminator</h3>
        <p className="mb-0">
          The discriminator evaluates the authenticity of generated images and their alignment with the input text descriptions.
          It uses a combination of image and text features to determine if an image-text pair is real or generated.
        </p>
      </div>
      
      <div className="card">
        <h2 className="mb-2">Research and Development</h2>
        <p>
          This project builds upon recent advances in text-to-image synthesis, including AttnGAN, StackGAN, and DALL-E.
          We've enhanced these approaches by incorporating BERT's contextual understanding of language to improve the
          semantic relevance of generated images.
        </p>
        <p className="mb-0">
          Our model is trained on large-scale datasets like MS-COCO, which contains images paired with multiple textual
          descriptions. This enables the model to learn the relationship between visual content and natural language.
        </p>
      </div>
    </div>
  );
}

export default AboutPage;