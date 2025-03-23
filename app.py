from flask import Flask, request, render_template, url_for, send_from_directory, redirect, flash
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import datetime
import random

# Add code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

# Import model components (with try/except to handle missing dependencies)
try:
    from code.miscc.config import cfg, cfg_from_file
    from code.miscc.model import G_NET
    from code.miscc.bert_encoder import BertEncoder
    MODEL_IMPORTS_SUCCESS = True
except Exception as e:
    print(f"Warning: Could not import model components: {e}")
    MODEL_IMPORTS_SUCCESS = False

app = Flask(__name__)
app.secret_key = 'kdgan-secret-key'

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/bird_KDGAN_hard.pth')
CONFIG_PATH = os.environ.get('CONFIG_PATH', 'code/cfg/eval_bird.yml')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'output/images'
DEMO_MODE = not os.path.exists(MODEL_PATH) or not MODEL_IMPORTS_SUCCESS

# Initialize models
text_encoder = None
netG = None

def load_models():
    """Load the text encoder and generator models"""
    global text_encoder, netG, DEMO_MODE  # Declare DEMO_MODE as global here
    
    if DEMO_MODE:
        print("Running in DEMO MODE - models will not be loaded")
        return
    
    try:
        print(f"Loading configuration from {CONFIG_PATH}...")
        cfg_from_file(CONFIG_PATH)
        
        print(f"Loading models on {DEVICE}...")
        
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
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        DEMO_MODE = True  # Now this is allowed because DEMO_MODE is declared as global
        print("Falling back to DEMO MODE")

# Rest of the code remains unchanged...