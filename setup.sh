#!/bin/bash

# Create necessary directories
mkdir -p data/birds
mkdir -p data/coco
mkdir -p models
mkdir -p output/images
mkdir -p DAMSMencoders/bird
mkdir -p DAMSMencoders/coco

# Download bird dataset
cd data/birds
curl http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xvzf CUB_200_2011.tgz
cd ../..

# Download COCO dataset
cd data/coco
curl http://images.cocodataset.org/zips/train2014.zip
 http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip
mv train2014 images
cp val2014/* images
cd ../..

# Download pretrained models
python google_drive.py 0B3y_msrWZaXLMzNMNWhWdW0zVWs eval/IS/bird/inception_finetuned_models.zip
python google_drive.py 1747il5vnY2zNkmQ1x_8hySx537ZAJEtj eval/FID/bird_val.npz
python google_drive.py 10NYi4XU3_bLjPEAg5KQal-l8A_d8lnL5 eval/FID/coco_val.npz

# Setup frontend
cd kdgan-frontend
npm install
cd ..

# Install Python requirements
pip install -r requirements.txt

echo "Setup completed successfully!"













# #!/bin/bash

# # Create necessary directories
# mkdir -p data/birds
# mkdir -p data/coco
# mkdir -p models
# mkdir -p output/images
# mkdir -p static/images
# mkdir -p code/cfg
# mkdir -p code/miscc
# mkdir -p eval/FID
# mkdir -p eval/IS/bird/inception/slim
# mkdir -p eval/IS/coco

# # Create placeholder image for demo mode
# echo "Creating placeholder image..."
# python -c "
# from PIL import Image, ImageDraw, ImageFont
# import os

# # Create directory if it doesn't exist
# os.makedirs('static/images', exist_ok=True)

# # Create a colored background
# img = Image.new('RGB', (512, 512), color=(70, 130, 180))
# draw = ImageDraw.Draw(img)

# # Add text
# draw.rectangle([10, 10, 502, 100], fill=(0, 0, 0, 128))
# try:
#     font = ImageFont.truetype('arial.ttf', 20)
# except:
#     font = ImageFont.load_default()

# draw.text((20, 20), 'KD-GAN Demo', fill=(255, 255, 255), font=font)
# draw.text((20, 50), 'Placeholder Image', fill=(255, 255, 255), font=font)

# # Save the image
# img.save('static/images/placeholder.jpg')
# print('Placeholder image created at static/images/placeholder.jpg')
# "

# # Install Python requirements
# echo "Installing Python requirements..."
# pip install torch torchvision transformers numpy pillow matplotlib tqdm nltk scikit-learn scipy easydict flask requests

# echo "Setup completed successfully!"
# echo "Note: This is a minimal setup for demo purposes."
# echo "To run the application, use: python app.py"