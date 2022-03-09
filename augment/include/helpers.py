import numpy as np
import librosa
#import matplotlib.pyplot as plt
from sklearn import metrics 
import os
import pickle
import time
import struct




""" 
    Data processing
"""


# Generates/extracts Log-MEL Spectrogram coefficients with LibRosa 
def get_mel_spectrogram_old(file_path, mfcc_max_padding=0, n_fft=2048, hop_length=512, n_mels=128):
    try:
        # Load audio file
        y, sr = librosa.load(file_path)

        # Normalize audio data between -1 and 1
        normalized_y = librosa.util.normalize(y)

        # Generate mel scaled filterbanks
        mel = librosa.feature.melspectrogram(normalized_y, sr=sr, n_mels=n_mels)

        # Convert sound intensity to log amplitude:
        mel_db = librosa.amplitude_to_db(abs(mel))

        # Normalize between -1 and 1
        normalized_mel = librosa.util.normalize(mel_db)

        # Should we require padding
        shape = normalized_mel.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
            xDiff = mfcc_max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            normalized_mel = np.pad(normalized_mel, pad_width=((0,0), (xLeft, xRight)), mode='constant')

    except Exception as e:
        print("Error parsing wavefile: ", e)
        return None 
    return normalized_mel



# Generates/extracts Log-MEL Spectrogram coefficients with LibRosa 
def get_mel_spectrogram(file_path, mfcc_max_padding=0, n_fft=2048, hop_length=512, n_mels=128):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=16000)

        # Normalize audio data between -1 and 1
        normalized_y = librosa.util.normalize(y)
        missing_values = 16000*4 - len(normalized_y)
        # print("signal len ", len(data), "pad len ", missing_values)
        if missing_values < 0:
            normalized_y = normalized_y[:64000]
        elif missing_values > 0:
            normalized_y = np.pad(
                normalized_y,
                (int(np.ceil(missing_values*0.5)), int(np.floor(missing_values*0.5)))
            )

        # Generate mel scaled filterbanks
        mel = librosa.feature.melspectrogram(normalized_y, sr=sr, n_mels=n_mels)

        # Convert sound intensity to log amplitude:
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize between -1 and 1
        normalized_mel = librosa.util.normalize(mel_db)


    except Exception as e:
        print("Error parsing wavefile: ", e)
        return None 
    return normalized_mel



# Generates/extracts MFCC coefficients with LibRosa 
def get_mfcc(file_path, mfcc_max_padding=0, n_mfcc=40):
    print(file_path)
    try:
        # Load audio file
        y, sr = librosa.load(file_path)
        print('file loaded')
        # Normalize audio data between -1 and 1
        normalized_y = librosa.util.normalize(y)

        # Compute MFCC coefficients
        mfcc = librosa.feature.mfcc(y=normalized_y, sr=sr, n_mfcc=n_mfcc)

        # Normalize MFCC between -1 and 1
        normalized_mfcc = librosa.util.normalize(mfcc)

        # Should we require padding
        shape = normalized_mfcc.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
            xDiff = mfcc_max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            normalized_mfcc = np.pad(normalized_mfcc, pad_width=((0,0), (xLeft, xRight)), mode='constant')

    except Exception as e:
        print("Error parsing wavefile: ", e)
        return None 
    return normalized_mfcc


# Given an numpy array of features, zero-pads each ocurrence to max_padding
def add_padding(features, mfcc_max_padding=174):
    padded = []

    # Add padding
    for i in range(len(features)):
        px = features[i]
        size = len(px[0])
        # Add padding if required
        if (size < mfcc_max_padding):
            xDiff = mfcc_max_padding - size
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            px = np.pad(px, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        
        padded.append(px)

    return padded


# Scales data between x_min and x_max
def scale(X, x_min, x_max, axis=0):
    nom = (X-X.min(axis=axis))*(x_max-x_min)
    denom = X.max(axis=axis) - X.min(axis=axis)
    denom[denom==0] = 1
    return x_min + nom/denom 


def save_split_distributions(test_split_idx, train_split_idx, file_path=None):
    if (path == None):
        print("You must enter a file path to save the splits")        
        return false

    
    # Create split dictionary
    split = {}
    split['test_split_idx'] = test_split_idx
    split['train_split_idx'] = train_split_idx

    with open(file_path, 'wb') as file_pi:
        pickle.dump(split, file_pi)

    return file


def load_split_distributions(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    return [data['test_split_idx'], data['train_split_idx']]
  

def find_dupes(array):
    seen = {}
    dupes = []

    for x in array:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return len(dupes)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    return train_score, test_score


def model_evaluation_report(model, X_train, y_train, X_test, y_test, calc_normal=True):
    dash = '-' * 38

    # Compute scores
    train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Pint Train vs Test report
    print('{:<10s}{:>14s}{:>14s}'.format("", "LOSS", "ACCURACY"))
    print(dash)
    print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Training:", train_score[0], 100 * train_score[1]))
    print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Test:", test_score[0], 100 * test_score[1]))


    # Calculate and report normalized error difference?
    if (calc_normal):
        max_err = max(train_score[0], test_score[0])
        error_diff = max_err - min(train_score[0], test_score[0])
        normal_diff = error_diff * 100 / max_err
        print('{:<10s}{:>13.2f}{:>1s}'.format("Normal diff ", normal_diff, ""))



# Expects a NumPy array with probabilities and a confusion matrix data, retuns accuracy per class
def acc_per_class(np_probs_array):    
    accs = []
    for idx in range(0, np_probs_array.shape[0]):
        correct = np_probs_array[idx][idx].astype(int)
        total = np_probs_array[idx].sum().astype(int)
        acc = (correct / total) * 100
        accs.append(acc)
    return accs



def compute_confusion_matrix(y_true, 
               y_pred, 
               classes, 
               normalize=False):

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


