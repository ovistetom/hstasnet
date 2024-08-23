import os
import torch
import torchaudio
from torch.utils.data import Dataset


EXT = 'wav'
SAMPLE_RATE = 44100


class MUSDB18Dataset(Dataset):
    """
    MUSDB18 dataset.

    Args:
        root (str): The root directory of the dataset.
    """

    def __init__(self,
                 root:str,
                 subset:str, 
                 sources:list = ['bass', 'drums', 'other', 'vocals'],
                 ):
        
        self.root = root
        self.subset = subset
        self.sources = sources
        self.path = os.path.join(root, subset)
        self.songs = self._collect_songs()

    def _get_source(self, name, source):
        return os.path.join(self.path, name, f'{source}.{EXT}')

    def _load_sample(self, n:int):

        song = self.songs[n]

        wav_sources = []
        num_frames = None
        for source in self.sources + ['mixture']:

            track = self._get_source(song, source)
            wav, sr = torchaudio.load(track)

            if sr != SAMPLE_RATE:
                raise ValueError(f"Expected sample rate {SAMPLE_RATE}, but got {sr}")
            if num_frames is None:
                num_frames = wav.shape[-1]
            elif wav.shape[-1] != num_frames:
                raise ValueError("Value 'num_frames' does not match across sources")
            
            wav_sources.append(wav)
        
        wav_mixture = wav_sources.pop()
        wav_sources = torch.stack(wav_sources)
         
        return wav_mixture, wav_sources

    def _collect_songs(self):

        song_names = []
        for song_name in os.listdir(self.path):
            if not song_name.startswith('.'):
                song_names.append(song_name)

        return sorted(song_names)
    
    def __getitem__(self, n: int):
            """
            Load the n-th sample from the dataset.

            Args:
                n (int): The index of the sample to be loaded.

            Returns:
                wav_mixture (torch.Tensor): Tensor of waveforms, size [C, L].
                wav_sources (torch.Tensor): Tensor of waveforms, size [S, C, L].
            """
            return self._load_sample(n)    
    
    def __len__(self):
        return len(self.songs)  
  

if __name__ == '__main__':

    root = r"data\musdb18_preprocessed"
    subset = 'valid'
    sources = ['bass', 'drums', 'vocals']
    dataset = MUSDB18Dataset(root, subset, sources)
    wav_sources, wav_mixture = dataset[0]
    print(f'{wav_sources.size() = }')
    print(f'{wav_mixture.size() = }')
    