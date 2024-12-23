from matplotlib.pyplot import stem
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as tt
import os
import sys
import stempeg
import random
import shutil
from tqdm import tqdm

# Add the parent directory to the path.
sys.path.append(os.path.join(os.path.dirname((os.path.dirname(__file__)))))
sys.path.append(os.path.join(os.path.dirname((os.path.dirname(__file__))), 'data'))


# Define constants.
STEM_DICT = {0: 'mixture', 1: 'drums', 2: 'bass', 3: 'other', 4: 'vocals'}


def segment_audio_array(audio_array, segment_length_in_s=20.0, fade_length_in_s=1.0, sample_rate=44100):
    """ Segment an audio array into segments of a given length. Apply fade-in and fade-out to the segments.

    Args:
        audio_array (np.ndarray): The audio array to segment.
        segment_length_in_s (float, optional): The length of each segment in seconds. Defaults to 20.0.
        fade_length_in_s (float, optional): The length of the fade in and out in seconds. Defaults to 1.0.
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.

    Returns:
        segments (np.ndarray): The segmented audio array.
    """
    
    assert audio_array.ndim == 3, f"Expected 3D array, got {audio_array.ndim}D array."

    # Segment the audio array.
    num_stems, num_channels, audio_length = audio_array.shape
    segment_length = int(sample_rate * segment_length_in_s)
    num_segments = audio_length // segment_length

    segments = np.zeros((num_stems, num_segments, num_channels, segment_length))

    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segments[:, i, :, :] = audio_array[:, :, start:end]

    # Apply fade-in and fade-out.
    fade_length = int(sample_rate * fade_length_in_s)
    segments = torch.from_numpy(segments)
    fade_transform = tt.Fade(fade_in_len=fade_length, fade_out_len=fade_length, fade_shape='linear')
    faded_segments = fade_transform(segments)
    segments = faded_segments.numpy()

    return segments


def preprocess_musdb18(database_path_src, database_path_dst, test_subset, hq=False):
    """ Preprocess the MUSDB18 dataset.

    Args:
        database_path_src (str): Path to the MUSDB18 dataset (with STEM files).
        database_path_dst (str): Path to the preprocessed MUSDB18 dataset (with WAV files).
        test_subset (list[str]): List of the tracks in the `test` subset.
        hq (bool, optional): Whether to use MUSDB18HQ version. Defaults to False.
    
    Returns:
        database_path_dst (str): Path to the preprocessed MUSDB18 dataset (with WAV files).
    """

    # Create the destination folder if it doesn't exist.
    os.makedirs(database_path_dst, exist_ok=True)

    # Iterate over the folders in the source folder.
    for subset_name in ['test', 'train']:

        subset_path_src = os.path.join(database_path_src, subset_name)
        song_list = os.listdir(subset_path_src)

        # Iterate over the files in the folder.
        for k, song_name in enumerate(tqdm(song_list, f"Processing '{subset_name}' set")):

            file_path_src = os.path.join(subset_path_src, song_name)

            # Split the files into training, test and validation sets.
            if subset_name == 'train':
                subset_path_dst = os.path.join(database_path_dst, 'train')
            elif song_name in test_subset:
                subset_path_dst = os.path.join(database_path_dst, 'test')
            else:
                subset_path_dst = os.path.join(database_path_dst, 'valid')
            os.makedirs(subset_path_dst, exist_ok=True)

            if not hq:
                # Check if the file is a .stem.mp4 file.
                if song_name.endswith('.stem.mp4'):

                    # Load the STEM file.
                    stems_array, sr = stempeg.read_stems(file_path_src)
                    stems_array = stems_array.transpose(0, 2, 1)
        
                    # Segment the audio array.
                    segments = segment_audio_array(stems_array)

                    # Re-create the mixture file by summing the sources.
                    segments[0] = segments[1:].sum(axis=0)

                    # Iterate over the stems (sources or mixture).
                    for i, stem_i in enumerate(segments):

                        # Iterate over the segments.
                        for j, segment_j in enumerate(stem_i):
                        
                            # Write the segment to the destination folder.
                            song_path_dst = os.path.join(subset_path_dst, f'track_{k:03}{j:02}')
                            os.makedirs(song_path_dst, exist_ok=True)
                            file_path_dst = os.path.join(song_path_dst, f'{STEM_DICT[i]}.wav')
                            sf.write(file_path_dst, data=segment_j.T, samplerate=sr, format='wav')

            else:

                # Load the WAV stem files into an array.
                stems_tensor_list = []
                for i in range(5):
                    song_path = os.path.join(subset_path_src, song_name)
                    source_i_file_name = os.path.join(song_path, f'{STEM_DICT[i]}.wav')
                    source_i_tensor, sr = torchaudio.load(source_i_file_name)
                    stems_tensor_list.append(source_i_tensor)
                stems_tensor = torch.stack(stems_tensor_list)
                stems_array = stems_tensor.numpy()
    
                # Segment the audio array.
                segments = segment_audio_array(stems_array)

                # Re-create the mixture file by summing the sources.
                segments[0] = segments[1:].sum(axis=0)

                # Iterate over the stems (sources or mixture).
                for i, stem_i in enumerate(segments):

                    # Iterate over the segments.
                    for j, segment_j in enumerate(stem_i):
                    
                        # Write the segment to the destination folder.
                        song_path_dst = os.path.join(subset_path_dst, f'track_{k:04}{j:02}')
                        os.makedirs(song_path_dst, exist_ok=True)
                        file_path_dst = os.path.join(song_path_dst, f'{STEM_DICT[i]}.wav')
                        sf.write(file_path_dst, data=segment_j.T, samplerate=sr, format='wav')                

    print(f"Successfully preprocessed MUSDB18 dataset into '{database_path_dst}'.")
    return database_path_dst


def data_augmentation_musdb18(database_path_src, database_path_dst, augmentation_ratio=2):
    """ Apply data augmentation to the MUSDB18 dataset, creating new cacophonic audio files.


    Args:
        database_path_src (str): Path to the preprocessed MUSDB18 dataset (with WAV files).
        database_path_dst (str): Path to the data-augmented MUSDB18 dataset (with WAV files).
        augmentation_ratio (int): Ratio of new audio files to create (e.g. '2' doubles the amount of files per subset).

    Returns:
        database_path_dst (str): Path to the data-augmented MUSDB18 dataset (with WAV files).
    """

    # Create the destination folder if it doesn't exist.
    os.makedirs(database_path_dst, exist_ok=True)    

    # Iterate over the folders in the source folder.
    for subset_name in ['test', 'train', 'valid']:

        subset_path_src = os.path.join(database_path_src, subset_name)
        subset_path_dst = os.path.join(database_path_dst, subset_name)
        track_list = os.listdir(subset_path_src)
        track_list_length = len(track_list)
        
        # Copy the contents of 'subset_path_src' to the 'subset_path_dst' directory.
        for file_name in track_list:
            file_path_src = os.path.join(subset_path_src, file_name)
            file_path_dst = os.path.join(subset_path_dst, file_name)
            shutil.copytree(file_path_src, file_path_dst, dirs_exist_ok=True)
    
        # Loop until the desired amount of files have been created.
        for k in tqdm(range((augmentation_ratio-1)*track_list_length), desc=f'Creating new files for {subset_name} subset.'):

            # Define list of stem tensors.
            stems_tensor_list = []

            # Loop to fetch four sources (and placeholder mixture).
            for i in range(5):
                random_track_name = random.choice(track_list)
                random_track_path = os.path.join(subset_path_src, random_track_name)
                source_i_file_name = os.path.join(random_track_path, f'{STEM_DICT[i]}.wav')
                source_i_tensor, sr = torchaudio.load(source_i_file_name)
                stems_tensor_list.append(source_i_tensor)
            
            # Stack the source tensors.
            stems_tensor = torch.stack(stems_tensor_list)     
            
            # Re-create the mixture file by summing the sources.
            stems_tensor[0] = stems_tensor[1:].sum(axis=0)

            # Write the destination folder.
            track_path_dst = os.path.join(subset_path_dst, f'track_1{k:03}00')
            os.makedirs(track_path_dst, exist_ok=True)

            # Iterate over the stems and write the corresponding stem WAV file.
            for i, stem_i in enumerate(stems_tensor):
                file_path_dst = os.path.join(track_path_dst, f'{STEM_DICT[i]}.wav')
                sf.write(file_path_dst, data=stem_i.numpy().T, samplerate=sr, format='wav')
            

    print(f"Successfully augmented MUSDB18 dataset into '{database_path_dst}'.")

    return database_path_dst  


def split_test_and_valid(database_path, subset_size=20):
    """ Split the MUSDB18 test set into test and validation subsets.
    
    Args:
        database_path (str): Path to the MUSDB18 dataset.
        subset_size (int, optional): Desired size of the 'test' subset. Defaults to 20.
    
    Returns:
        subset (list[str]): Subset of file names for the MUSDB18 'test' subset.
    """

    database_test_path_src = os.path.join(database_path, 'test')
    database_test_list = os.listdir(database_test_path_src)
    test_subset = random.sample(database_test_list, subset_size)

    print(f"Successfully split test set (test size = {subset_size}, valid size = {len(database_test_list) - subset_size}).")

    return test_subset


if __name__ == '__main__':

    # database_path_src = "data/musdb18hq"
    # database_path_dst = "data/musdb18hq_preprocessed"

    # test_subset = split_test_and_valid(database_path_src, subset_size=20)
    # preprocess_musdb18(database_path_src, database_path_dst, test_subset, hq=True)

    data_augmentation_musdb18(database_path_src="data/musdb18hq_preprocessed", database_path_dst="data/musdb18hq_augmented", augmentation_ratio=4)
