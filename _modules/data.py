from curses import meta
from numpy import int64, array
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Tuple
import librosa
import numpy as np
from itertools import compress
from pathlib import Path
from random import shuffle
from math import floor
from torchvision.transforms import Lambda, RandomApply, RandomChoice
from sklearn.preprocessing import LabelEncoder
from random import shuffle
from math import floor
from pandas import read_csv
from random import shuffle
import soundfile as sf
import librosa
from tqdm import tqdm

class AudioDataset(Dataset):
    
    def __init__(self,
                 file_path: Path = Path("./"),
                 feat: str = "aug", ##Options raw|aug   ###folderList va rivisto....
                 dataset: str = "MAVD", ##Oprions MAVD|UrbanSound8K|GRN
                 folderList: list = list(range(1, 11))) \
                     -> None:
        ###IN REALTA' NON VIENE MAI PASSATA UNA LISTA COMUNQUE LASCIAMO PURE COSI
        #initialize lists to hold file names, labels, and folder numbers
        self.file_path = str(file_path)
        self.parent_path = file_path.joinpath(feat)
        self.dataset = dataset        
        self.feat = feat
        #self.file_names = sorted(file_path.iterdir())
        
        self.folds = [self._read_element(i) for i in folderList]
        
        self.X_data = [f[0] for f in self.folds]
        self.X_data = np.concatenate(self.X_data, axis=0)
        
        self.y_data = [f[1] for f in self.folds]
        self.y_data = np.concatenate(self.y_data, axis=0)
        print(self.y_data.shape)
        print(self.X_data.shape)
        
    def _read_element(self,
                      index: str) -> Tuple[Tensor, int64]:
        metadata = read_csv(self.file_path+"/metadata/"+self.dataset+"_augmented.csv")
        if self.dataset == 'UrbanSound8K':
            index = int(index)
        if self.feat == "aug":
            
            X = np.load(self.parent_path.joinpath(f"X-40mel_spec_v2_aug.npy"))
            y = np.load(self.parent_path.joinpath(f"y-40mel_spec_v2_aug.npy"))
            split_idx = metadata.index[metadata['fold'] == index].tolist()
            shuffle(split_idx)
            X = np.take(X, split_idx, axis=0)
            y = np.take(y, split_idx, axis=0)
        elif self.feat == "raw":
            #metadata = read_csv(self.file_path+"/metadata/"+self.dataset+".csv")
            metadata=metadata.loc[lambda metadata: metadata['augment']=='none']
            X = np.load(self.parent_path.joinpath(f"X-40mel_spec_v2.npy"))
            y = np.load(self.parent_path.joinpath(f"y-40mel_spec_v2.npy"))
            
            split_idx = metadata.index[metadata['fold'] == index].tolist()
            shuffle(split_idx)
            
            X = np.take(X, split_idx, axis=0)
            y = np.take(y, split_idx, axis=0)
            
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded
    
    def __getitem__(self, 
                    index: int) \
                    -> Tuple[Tensor, int64]:
        #print(index)
        #print(self.X_data[index])
        #print(self.y_data[index])
        return self.X_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.X_data)


class US8k(LightningDataModule):
    def __init__(self,
                 batch_size: int = 64,
                 file_path: str = "/UrbanSound8K/",
                 transforms: bool = False,
                 debug: bool = False,
                 augmentation: bool = True,
                 waveform: bool = False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.file_path = file_path
        self.transforms = transforms
        self.augmentation = augmentation
        self.debug = debug
        self.waveform = waveform

        self.dl_params = {"batch_size": batch_size,
                          "num_workers": 4,
                          "persistent_workers": True,
                          "pin_memory": False}
        if self.augmentation:
            augtype='aug'
        else:
            augtype='raw'
        
        if not self.waveform:
            self.folds = [
                AudioDataset(
                    Path(self.file_path),
                    augtype,
                    "UrbanSound8K",
                    [str(i)],
                ) for i in range(1, 11)
            ]
            
            self.test_folds = [
                AudioDataset(
                    Path(self.file_path),
                    "raw",
                    'UrbanSound8K',
                    [str(i)],
                ) for i in range(1, 11)
            ]
        else:
            self.folds = [
                AudioDataset1D(
                    Path(self.file_path),
                    augtype,
                    "UrbanSound8K",
                    [str(i)],
                ) for i in range(1, 11)
            ]
            
            self.test_folds = [
                AudioDataset1D(
                    Path(self.file_path),
                    "raw",
                    'UrbanSound8K',
                    [str(i)],
                ) for i in range(1, 11)
            ]
            

    def shuffle_folds(self,
                      testfold_id: int) -> None:
        mask = [1] * 10
        val_index = testfold_id - 1 if (testfold_id - 1) > 0 else (len(self.folds) - 1)
        mask[val_index] = 0
        mask[testfold_id] = 0
        self.train_set = list(compress(self.folds, mask))
        self.val_set = self.test_folds[val_index]
        self.test_set = self.test_folds[testfold_id]

    def setup(self, stage=None):
        pass
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(ConcatDataset(self.train_set), shuffle=True, **self.dl_params)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, **self.dl_params)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, **self.dl_params)


class GRN(LightningDataModule):
    def __init__(self,
                 batch_size: int = 64,
                 file_path: str = "GRN/",
                 transforms: bool = False,
                 augmentation: bool = True,
                 debug: bool = False,
                 waveform: bool = False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.file_path = file_path
        self.transforms = transforms
        self.augmentation = augmentation
        self.debug = debug
        self.waveform = waveform
        
        self.dl_params = {"batch_size": batch_size,
                          "num_workers": 4,
                          "persistent_workers": True,
                          "pin_memory": False}
        if self.augmentation:
            augtype = 'aug'
        else:
            augtype = 'raw'
        self.folds = [
            AudioDataset(
                Path(self.file_path),
                augtype,
                "GRN",
                ['fold'+str(i)],
            ) for i in range(0, 10)
        ]
        
        self.test_folds = [
            AudioDataset(
                Path(self.file_path),
                "raw",
                'GRN',
                ['fold'+str(i)],
            ) for i in range(0, 10)
        ]

    def shuffle_folds(self,
                      testfold_id: int) -> None:
        mask = [1] * 10
        val_index = testfold_id - 1 if (testfold_id - 1) > 0 else (len(self.folds) - 1)
        mask[val_index] = 0
        mask[testfold_id] = 0
        self.train_set = list(compress(self.folds, mask))
        self.val_set = self.test_folds[val_index]
        self.test_set = self.test_folds[testfold_id]

    def setup(self, stage=None):
        pass
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(ConcatDataset(self.train_set), shuffle=True, **self.dl_params)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, **self.dl_params)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, **self.dl_params)


class MAVD(LightningDataModule):
    def __init__(self,
                 batch_size: int = 1,
                 file_path: str = "MAVD/",
                 transforms: bool = False,
                 augmentation: bool = True,
                 debug: bool = True,
                 waveform: bool = False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.file_path = file_path
        self.transforms = transforms
        self.augmentation = augmentation
        self.debug = debug
        self.waveform = waveform
        
        self.dl_params = {"batch_size": batch_size,
                          "num_workers": 4,
                          "persistent_workers": True,
                          "pin_memory": False}
        if self.augmentation:
            augtype='aug'
        else:
            augtype = 'raw'
            
        self.train_set=AudioDataset(
                Path(self.file_path),
                augtype,
                "MAVD",
                ['train'])
        
        self.test_set = AudioDataset(
                Path(self.file_path),
                "raw",
                'MAVD',
                ['test']
            )
        
        self.valid_set = AudioDataset(
                Path(self.file_path),
                "raw",
                'MAVD',
                ['validate']
            )

    #def shuffle_folds(self) -> None:
    #    mask = [1] * 10
    #    val_index = testfold_id - 1 if (testfold_id - 1) > 0 else (len(self.folds) - 1)
    #    mask[val_index] = 0
    #    mask[testfold_id] = 0
    #    self.train_set = list(compress(self.folds, mask))
    #    self.val_set = self.test_folds[val_index]
    #    self.test_set = self.test_folds[testfold_id]

    def setup(self, stage=None):
        pass
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, shuffle=True, **self.dl_params)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, **self.dl_params)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, **self.dl_params)


def _pitch_shift(waveform: array,
                 pitch_step: list = None,    # usually from [-2,-1, 1, 2]
                 noise_level: float = None,
                 sr: int = 8000) \
                     -> Tuple[array, array, int]:
    """Performs pitch shift for data augmentation"""
    wav_data_augmented = waveform    
    if pitch_step != None:
        wav_data_augmented = librosa.effects.pitch_shift(waveform, sr, n_steps=pitch_step)

    if noise_level != None:
        wav_data_augmented = waveform  + np.random.normal(scale=noise_level, size=waveform.shape)
        
    return wav_data_augmented

    
pitch_shift_transform = RandomChoice([
    Lambda(lambda x: _pitch_shift(x, pitch_step=-2, noise_level=0.1)),
    Lambda(lambda x: _pitch_shift(x, pitch_step=-1, noise_level=0.1)),
    Lambda(lambda x: _pitch_shift(x, pitch_step=1, noise_level=0.1)),
    Lambda(lambda x: _pitch_shift(x, pitch_step=2, noise_level=0.1))]
)

us8k_transform = RandomApply([pitch_shift_transform], p=4/5)

class AudioDataset1D(Dataset):

    def __init__(self,
                 file_path: Path = Path("/UrbanSound8K"),
                 feat: str = "raw", ##Options raw|aug   ###folderList va rivisto....
                 dataset: str = "UrbanSound8K", ##Oprions MAVD|UrbanSound8K|GRN
                 folderList: list = list(range(1, 11))) \
                     -> None:
        print(f"Loading fold(s): {folderList}")
        #initialize lists to hold file names, labels, and folder numbers
        self.file_path = str(file_path)
        self.parent_path = file_path.joinpath(feat)
        self.dataset = dataset        
        self.feat = feat
        #self.file_names = sorted(file_path.iterdir())

        self.folds = [self._read_fold(i) for i in tqdm(folderList)]
        
        self.X_data = [f[0] for f in self.folds]
        self.X_data = np.concatenate(self.X_data, axis=0)
        
        self.y_data = [f[1] for f in self.folds]
        self.y_data = np.concatenate(self.y_data, axis=0)
        
        # print(self.y_data.shape)
        # print(self.X_data.shape)
    
    def _read_element(self, index, fname):
        audio = sf.read(fname)[0]
        if len(audio.shape) >= 2:
            audio = np.mean(audio, axis=1)
            
        audio = librosa.util.normalize(audio)
            
        missing_values = 64000 - len(audio)
        # print("signal len ", len(data), "pad len ", missing_values)
        if missing_values < 0:
            audio = audio[:64000]
        elif missing_values > 0:
            audio = np.pad(
                audio,
                (int(np.ceil(missing_values*0.5)), int(np.floor(missing_values*0.5)))
            )
    
        return us8k_transform(audio)
    
    def _read_fold(self,
                   index: str) -> Tuple[Tensor, int64]:
        index = int(index)
        if self.feat == "aug":
            metadata = read_csv(self.file_path+"/metadata/"+self.dataset+"_augmented.csv")
        else:
            metadata = read_csv(self.file_path+"/metadata/"+self.dataset+".csv")
        
        X = [self._read_element(index, self.file_path + f"/audio/fold{index}/" + f) for f in metadata[metadata["fold"] == index].slice_file_name.to_list()]
        y = metadata[metadata["fold"] == index].classID.to_list()

        X = np.array(X)
        y = np.array(y)
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(y_encoded.shape)
                
        return X, y
    
    def __getitem__(self, 
                    index: int) \
                    -> Tuple[Tensor, int64]:
        return self.X_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.X_data)


if __name__ == "__main__":
    a = AudioDataset1D(folderList=[1])
    
    from dnet import FilterBankConv1d
    import torch
    
    x = np.expand_dims(a[0][0], [0, 1]) #, FilterBankConv1d(320, 2)(a[0][0]).shape)
    print(x.shape)
    
    l = FilterBankConv1d(320, 2)
    g = torch.nn.Conv1d(8, 16, kernel_size=64, stride=4)
    print(g(l(Tensor(x))).shape)