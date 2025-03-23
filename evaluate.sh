#!/bin/bash

# Generate images
cd code
python main.py --cfg cfg/eval_bird.yml --gpu 0
python main.py --cfg cfg/eval_coco.yml --gpu 0
cd ..

# Calculate Inception Score for bird
cd eval/IS/bird
python inception_score_bird.py ../../../models/bird_KDGAN_hard
cd ../../..

# Calculate Inception Score for COCO
cd eval/IS/coco
python inception_score_coco.py ../../../models/coco_KDGAN_hard
cd ../../..

# Calculate FID for bird
cd eval/FID
python fid_score.py --gpu 0 --batch-size 50 --path1 bird_val.npz --path2 ../../models/bird_KDGAN_hard
cd ../..

# Calculate FID for COCO
cd eval/FID
python fid_score.py --gpu 0 --batch-size 50 --path1 coco_val.npz --path2 ../../models/coco_KDGAN_hard
cd ../..

echo "Evaluation completed successfully!"