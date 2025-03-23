#!/bin/bash

# Train bird model
cd code
python main.py --cfg cfg/bird_KDGAN.yml --gpu 0
cd ..

# Train COCO model
cd code
python main.py --cfg cfg/coco_KDGAN.yml --gpu 0
cd ..

echo "Training completed successfully!"