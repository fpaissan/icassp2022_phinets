
# Required libraries
import sys
import os
import pandas as pd
import numpy as np
import librosa
import pickle
from include import helpers
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
import argparse


parser = argparse.ArgumentParser(description='Extract features from an audio dataset. Argument: dataset name boolean augmented features or not')
parser.add_argument('-d', '--dataset', type=str, required = True, help='Dataset name (folder)', choices=['MAVD','UrbanSound8K','GRN'])
parser.add_argument('-a', '--augment', default=False, action='store_true')
args = parser.parse_args()
dataset = args.dataset
augment = args.augment
print('Extracting features for dataset %s'%dataset)
if augment:
        print('Using agumented features where available')
else:
        print('Using original features')

##Parameterize output folder???

#dataset='MAVD' #'UrbanSound8K'
#dataset='UrbanSound8K'
#augment=False ##True

audio_path = os.path.abspath(dataset+'/audio')
metadata_path = os.path.abspath(dataset+'/metadata/'+dataset+'_augmented.csv')

# Load the metadata from the generated CSV
metadata = pd.read_csv(metadata_path)
if not augment:
    metadata=metadata.loc[lambda metadata: metadata['augment']=='none']
# Examine dataframe
print("Metadata length:", len(metadata))

# Iterate through all audio files and extract Log-Mel Spectrograms

features = []
labels = []
frames_max = 0
counter = 0
total_samples = len(metadata)
n_mels = 40

for index, row in metadata.iterrows():
    if dataset == 'UrbanSound8K':
        if str(row["augment"])=='none':
            file_path = os.path.join(audio_path, 'fold' + str(row["fold"]), str(row["file"]))
        else:
            file_path = os.path.join(audio_path+'_augmented', 'fold' + str(row["fold"]), str(row["augment"]), str(row["file"]))
    else:
        file_path = row['file']
        
    class_label = row["class"]

    # Extract Log-Mel Spectrograms (do not add padding)
    mels = helpers.get_mel_spectrogram(file_path, 0, n_mels=n_mels)
    
    # Save current frame count
    num_frames = mels.shape[1]
    
    # Add row (feature / label)
    features.append(mels)
    labels.append(class_label)

    # Update frames maximum
    if (num_frames > frames_max):
        frames_max = num_frames
        
    print("Progress: {}/{}".format(index+1, total_samples), end='\r')

    counter += 1
    
print("Finished: {}/{}".format(index, total_samples))

padded_features = helpers.add_padding(features, frames_max)

X = np.array(padded_features)
y = np.array(labels)

print(x.shape)
print(y.shape)
# Optionally save the features to disk
####Check folder exists or create
if augment:
    outfolder=dataset+'/aug'
else:
    outfolder=dataset+'/raw'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
if augment:
    np.save(outfolder+"/X-40mel_spec_v2_aug", X)
    np.save(outfolder+"/y-40mel_spec_v2_aug", y)
else:
    np.save(outfolder+"/X-40mel_spec_v2", X)
    np.save(outfolder+"/y-40mel_spec_v2", y)
# Verify shapes
print("Raw features length: {}".format(len(features)))
print("Padded features length: {}".format(len(padded_features)))
print("Feature labels length: {}".format(len(features)))
print("X: {}, y: {}".format(X.shape, y.shape))


