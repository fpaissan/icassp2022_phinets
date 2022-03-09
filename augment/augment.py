import os
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
import random
import argparse


parser = argparse.ArgumentParser(description='Augmentation of an audio dataset applyint time streching, pitch shifting and additive noise. Argument: dataset name')
parser.add_argument('-d', '--dataset', type=str, required = True, help='Dataset name (folder)', choices=['MAVD','UrbanSound8K','GRN'])
args = parser.parse_args()
dataset = args.dataset
# Set your path to the dataset
#dataset='MAVD' ##'UrbanSound8K'
#dataset='UrbanSound8K'

data_path = os.path.abspath(dataset)

audio_path = os.path.join(data_path, 'audio')
augmented_path = os.path.join(data_path, 'audio_augmented')

# Metadata
metadata_augmented_path = os.path.join(data_path,'metadata/'+dataset+'_augmented.csv')
metadata_path = os.path.join(data_path,'metadata/'+dataset+'.csv')


# Load the metadata from the generated CSV

if dataset == 'MAVD':
        metadata_ori = pd.read_csv(metadata_path)
        metadata = metadata_ori.loc[lambda metadata: metadata['Fold']=='train'] ##IF MAVD IT APPLYS THE AUGMENTATION ONLY ON THE TRAIN SET
else:
        metadata = pd.read_csv(metadata_path)
###Time stretching
rates = [0.81, 1.07]
total = len(metadata) * len(rates)
count = 0

##Lists for new metadata
aug_names=[]
aug_classes=[]
aug_class_id=[]
aug_folds=[]
aug_augmentations=[]
for rate in rates:
        # Generate new stretched audio file
        for index, row in metadata.iterrows():
                if dataset == 'UrbanSound8K':
                        curr_file_path = audio_path + '/fold' + str(row['fold']) + '/' + row['slice_file_name']
                        curr_rate_path = augmented_path + '/fold' + str(row['fold']) + '/speed_' + str(int(rate*100))
                        output_path = curr_rate_path + '/' + row['slice_file_name']
                else:
                        curr_file_path = audio_path + '/' + row['Fold'] + '/' + row['File'].split('/')[-1]
                        curr_rate_path = augmented_path + '/' + row['Fold'] + '/speed_' + str(int(rate*100))
                        output_path = curr_rate_path + '/' + row['File'].split('/')[-1]
                        aug_names.append(output_path)
                        aug_classes.append(row['Event'])
                        aug_class_id.append(row['ClassID'])
                        aug_folds.append(row['Fold'])
                        aug_augmentations.append('speed_' + str(int(rate*100)))
        
                # Create sub-dir if it does not exist
                if not os.path.exists(curr_rate_path):
                        os.makedirs(curr_rate_path)
                    
        
                # Skip when file already exists
                if (os.path.isfile(output_path)):
                        count += 1 
                        continue
        
                y, sr = librosa.load(curr_file_path)  
                y_changed = librosa.effects.time_stretch(y, rate=rate)
                sf.write(output_path, y_changed, sr)
        
                count += 1 
        
                print("Rates Progress: {}/{}".format(count, total))
                print("Last file: ", curr_file_path.split('/')[-1])
                
tone_steps = [-1, -2, 1, 2]
total = len(metadata) * len(tone_steps)
count = 0
for tone_step in tone_steps:
    # Generate new pitched audio
        for index, row in metadata.iterrows():
                if dataset == 'UrbanSound8K':
                        curr_file_path = audio_path + '/fold' + str(row['fold']) + '/' + row['slice_file_name']
                        curr_ps_path = augmented_path + '/fold' + str(row['fold']) + '/pitch_' + str(tone_step)
                        output_path = curr_ps_path + '/' + row['slice_file_name']
                else:
                        curr_file_path = audio_path + '/' + row['Fold'] + '/' + row['File'].split('/')[-1]
                        curr_ps_path = augmented_path + '/' + row['Fold'] + '/pitch_' + str(tone_step)
                        output_path = curr_ps_path + '/' +row['File'].split('/')[-1]
                        aug_names.append(output_path)
                        aug_classes.append(row['Event'])
                        aug_class_id.append(row['ClassID'])
                        aug_folds.append(row['Fold'])
                        aug_augmentations.append('pitch_' + str(tone_step))
                # Create sub-dir if it does not exist
                if not os.path.exists(curr_ps_path):
                        os.makedirs(curr_ps_path)
                # Skip when file already exists
                if (os.path.isfile(output_path)):
                    count += 1 
                    continue
        
                y, sr = librosa.load(curr_file_path)  
                y_changed = librosa.effects.pitch_shift(y, sr, n_steps=tone_step)
                sf.write(output_path, y_changed, sr)
        
                count += 1 
        
                print("Pitch Progress: {}/{}".format(count, total))
                print("Last file: ", curr_file_path.split('/')[-1])


def add_noise(data):
    noise = np.random.rand(len(data))
    noise_amp = random.uniform(0.005, 0.008)
    data_noise = data + (noise_amp * noise)
    return data_noise

total = len(metadata)
count = 0

# Generate new noised audio
for index, row in metadata.iterrows():
        if dataset == 'UrbanSound8K':
                curr_file_path = audio_path + '/fold' + str(row['fold']) + '/' + row['slice_file_name']
                curr_noise_path = augmented_path + '/fold' + str(row['fold']) + '/noise'
                output_path = curr_noise_path + '/' + row['slice_file_name']
        else:
                curr_file_path = audio_path + '/' + row['Fold'] + '/' + row['File'].split('/')[-1]
                curr_noise_path = augmented_path + '/' + row['Fold'] + '/noise'
                output_path = curr_noise_path + '/' + row['File'].split('/')[-1]
                aug_names.append(output_path)
                aug_classes.append(row['Event'])
                aug_class_id.append(row['ClassID'])
                aug_folds.append(row['Fold'])
                aug_augmentations.append('noise')
        # Create sub-dir if it does not exist
        if not os.path.exists(curr_noise_path):
                os.makedirs(curr_noise_path)
        
        # Skip when file already exists
        if (os.path.isfile(output_path)):
                count += 1 
                continue
        
        y, sr = librosa.load(curr_file_path)  
        y_changed = add_noise(y)
        sf.write(output_path, y_changed, sr)
    
        count += 1 

        print("Noise Progress: {}/{}".format(count, total))
        print("Last file: ", curr_file_path.split('/')[-1])



def get_files_recursive(path):
    # create a list of file and sub directories names in the given directory 
    file_list = os.listdir(path)
    all_files = list()
    # Iterate over all the entries
    for entry in file_list:
        # Create full path
        full_path = os.path.join(path, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(full_path):
            all_files = all_files + get_files_recursive(full_path)
        else:
            all_files.append(full_path)
                
    return all_files   



if dataset == 'UrbanSound8K':
        # Get every single file within the tree
        files = get_files_recursive(augmented_path)
        # Define metadata columns
        names = []
        classes = []
        folds = []
        augmentations = []
        # Iterate and collect name, fold and class
        for file in files:
                pieces = file.split("/")
                file = pieces[len(pieces) - 1]
                fold = pieces[len(pieces) - 3] 
                augment = pieces[len(pieces) - 2] 
                fold_num = fold[4:len(fold)]
                class_id = file.split("-")[1]

                # Push records
                names.append(file)
                folds.append(fold_num)
                classes.append(class_id)
                augmentations.append(augment)

        # Create a dataframe with the new augmented data
        new_meta = pd.DataFrame({'file': names, 'fold': folds, 'class_id': classes, 'augment': augmentations })

        # Make sure class_id is int
        new_meta['class_id'] = new_meta['class_id'].astype(np.int64)

        print(len(new_meta), "new entries")

        # Add class names to the new dataframe using merge
        classes = pd.DataFrame({
                'class_id': range(0,10),
                'class': [
                        'air_conditioner',
                        'car_horn',
                        'children_playing',
                        'dog_bark',
                        'drilling',
                        'engine_idling',
                        'gun_shot',
                        'jackhammer',
                        'siren',
                        'street_music'
                ]
        })
        
        new_meta = pd.merge(new_meta, classes, on='class_id')
        new_meta.tail()

        # Modify original data to fit the new structure
        del metadata['fsID'], metadata['start'], metadata['end'], metadata['salience']
        metadata.columns = ['file', 'fold', 'class_id', 'class']
        metadata['augment'] = 'none'

        # Concat the two dataframes
        full_meta = pd.concat([metadata_ori, new_meta])

elif dataset == "MAVD":
        new_meta = pd.DataFrame({'file': aug_names, 'fold': aug_folds, 'class': aug_classes, 'class_id': aug_class_id, 'augment': aug_augmentations })
        
        # Modify original data to fit the new structure
        del metadata_ori['Start'], metadata_ori['End']
        metadata_ori.columns = ['index','file', 'fold', 'class_id', 'class']
        metadata_ori['augment'] = 'none'
        del metadata_ori['index']
        # Concat the two dataframes
        full_meta = pd.concat([metadata_ori, new_meta])
else:
        new_meta = pd.DataFrame({'file': aug_names, 'fold': aug_folds, 'class': aug_classes, 'class_id': aug_class_id, 'augment': aug_augmentations })
        del metadata['Start'], metadata['End']
        metadata.columns = ['index','file', 'fold', 'class_id', 'class']
        metadata['augment'] = 'none'
        del metadata['index']
        # Concat the two dataframes
        full_meta = pd.concat([metadata, new_meta])
# Verify lengths
if (len(full_meta) == len(metadata) + len(new_meta)):
        print("Dataframes merged correctly!")
else:
        print("Error! Lengths do not match.")
                
print("Initial data:", len(metadata))
print("New data:", len(new_meta))
print("Merged data:", len(full_meta))

# Save the new metadata
full_meta.to_csv(metadata_augmented_path, index=False, encoding="utf-8")
