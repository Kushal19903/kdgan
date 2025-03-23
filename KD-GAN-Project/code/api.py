from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import time
import sys
import yaml
from easydict import EasyDict as edict

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from miscc.config import cfg, cfg_from_file
from miscc.model import G_NET
from miscc.bert_encoder import BertEncoder
from datasets import prepare_data

app = Flask(__name__, static_folder='../output/images')

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', '../models/bird_KDGAN_hard.pth')
CONFIG_PATH = os.environ.get('CONFIG_PATH', 'cfg/eval_bird.yml')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
text_encoder = None
netG = None

def load_models():
    global text_encoder, netG
    
    print(f"Loading configuration from {CONFIG_PATH}...")
    cfg_from_file(CONFIG_PATH)
    
    print(f"Loading models from {MODEL_PATH}...")
    
    # Initialize models
    text_encoder = BertEncoder(cfg, DEVICE)
    text_encoder = text_encoder.to(DEVICE)
    
    netG = G_NET()
    netG = netG.to(DEVICE)
    
    # Load checkpoint if available
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        netG.load_state_dict(state_dict)
        print("Models loaded successfully!")
    else:
        print(f"Warning: Model checkpoint not found at {MODEL_PATH}")
        print("Using untrained models!")
    
    # Set models to evaluation mode
    netG.eval()

@app.before_request
def initialize():
    if text_encoder is None or netG is None:
        load_models()

@app.route('/api/generate', methods=['POST'])
def generate_image():
    try:
        # Get text from request
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Process text for the model
        # This is a simplified version - in a real implementation, you would need to
        # tokenize and process the text according to your model's requirements
        captions = [text]
        cap_lens = [len(text.split())]
        
        # Generate image
        with torch.no_grad():
            # Get text embedding
            word_features, sent_features = text_encoder(captions, cap_lens)
            
            # Sample noise
            noise = torch.randn(1, cfg.GAN.Z_DIM).to(DEVICE)
            
            # Generate image
            fake_imgs, _, _, _ = netG(noise, sent_features, word_features, None)
            
            # Get the highest resolution image
            fake_img = fake_imgs[-1][0]
            
            # Denormalize image
            fake_img = (fake_img + 1) / 2.0  # [-1, 1] -> [0, 1]
            fake_img = fake_img.permute(1, 2, 0).cpu().numpy()
            fake_img = (fake_img * 255).astype(np.uint8)
            
            # Convert to base64
            image = Image.fromarray(fake_img)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Save image to disk
            timestamp = int(time.time())
            filename = f"generated_{timestamp}.png"
            image_path = os.path.join('../output/images', filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.save(image_path)
            
            # Return image URL
            image_url = f"/static/images/{filename}"
            
            return jsonify({
                'success': True,
                'imageUrl': image_url,
                'imageBase64': f"data:image/png;base64,{img_str}",
                'text': text
            })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gallery', methods=['GET'])
def get_gallery():
    try:
        # Get all generated images
        images_dir = '../output/images'
        os.makedirs(images_dir, exist_ok=True)
        
        images = []
        for filename in os.listdir(images_dir):
            if filename.startswith('generated_') and filename.endswith('.png'):
                image_path = os.path.join(images_dir, filename)
                image_url = f"/static/images/{filename}"
                
                # Extract timestamp from filename
                timestamp = filename.replace('generated_', '').replace('.png', '')
                
                images.append({
                    'id': timestamp,
                    'url': image_url,
                    'caption': 'Generated image',  # You could store captions in a database
                    'timestamp': timestamp
                })
        
        # Sort by timestamp (newest first)
        images.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify(images)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('../output/images', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)