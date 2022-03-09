# Data augmentation and feature extraction
python augment/augment.py -d UrbanSound8K -p $1
python augment/extract_features.py -d UrbanSound8K -p $1 -a True

# Model training
python main.py -a UrbanSound8K cfgs $1