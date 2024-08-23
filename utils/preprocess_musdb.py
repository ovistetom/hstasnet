import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as tt
import os
import sys
import stempeg
import random
from tqdm import tqdm

# Add the parent directory to the path.
sys.path.append(os.path.join(os.path.dirname((os.path.dirname(__file__)))))
sys.path.append(os.path.join(os.path.dirname((os.path.dirname(__file__))), 'data'))


# Define constants.
STEM_DICT = {0: 'mixture', 1: 'drums', 2: 'bass', 3: 'other', 4: 'vocals'}


def segment_stem_array(stem_array, segment_length_in_s=20.0, fade_length_in_s=1.0, sample_rate=44100):
    """
    Segment an audio array into segments of a given length. Apply fade-in and fade-out to the segments.

    Args:
        stem_array (np.ndarray): The audio array to segment.
        segment_length_in_s (float, optional): The length of each segment in seconds. Defaults to 20.0.
        fade_length_in_s (float, optional): The length of the fade in and out in seconds. Defaults to 1.0.
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.

    Returns:
        segments (np.ndarray): The segmented audio array.
    """
    
    # Segment the audio array.
    num_stems, num_channels, audio_length = stem_array.shape
    segment_length = int(sample_rate * segment_length_in_s)
    num_segments = audio_length // segment_length

    segments = np.zeros((num_stems, num_segments, num_channels, segment_length))

    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segments[:, i, :, :] = stem_array[:, :, start:end]

    # Apply fade-in and fade-out.
    fade_length = int(sample_rate * fade_length_in_s)
    segments = torch.from_numpy(segments)
    fade_transform = tt.Fade(fade_in_len=fade_length, fade_out_len=fade_length, fade_shape='linear')
    faded_segments = fade_transform(segments)
    segments = faded_segments.numpy()

    return segments


def preprocess_musdb(database_path_src, database_path_dst, test_subset):
    """
    Preprocess the MUSDB18 dataset.

    Args:
        database_path_src (str): The path to the MUSDB18 dataset.
        database_path_dst (str): The path to the preprocessed MUSDB18 dataset.
    """

    # Create the destination folder if it doesn't exist.
    if not os.path.exists(database_path_dst):
        os.makedirs(database_path_dst)

    # Iterate over the folders in the source folder.
    for subset_name in ['test', 'train']:

        subset_path_src = os.path.join(database_path_src, subset_name)

        # Iterate over the files in the folder.
        for song_name in tqdm(os.listdir(subset_path_src), f"Processing '{subset_name}'"):

            file_path_src = os.path.join(subset_path_src, song_name)

            # Split the files into training, test and validation sets.
            if subset_name == 'train':
                subset_path_dst = os.path.join(database_path_dst, 'train')
            elif song_name in test_subset:
                subset_path_dst = os.path.join(database_path_dst, 'test')
            else:
                subset_path_dst = os.path.join(database_path_dst, 'valid')
            os.makedirs(subset_path_dst, exist_ok=True)

            # Check if the file is a .mp4 file.
            if song_name.endswith('.mp4'):

                song_name_dst = song_name[:-9]

                # Load the STEM file.
                stem_array, sr = stempeg.read_stems(file_path_src)
                stem_array = stem_array.transpose(0, 2, 1)
    
                # Process the STEM array.
                segments = segment_stem_array(stem_array)
                segments[0] = segments[1:].sum(axis=0)

                # Iterate over the sources.
                for i, stem_i in enumerate(segments):

                    # Iterate over the segments.
                    for j, segment_j in enumerate(stem_i):
                    
                        # Write the segment to the destination folder.
                        song_path_dst = os.path.join(subset_path_dst, f'{song_name_dst} ({j:02})')
                        os.makedirs(song_path_dst, exist_ok=True)
                        file_path_dst = os.path.join(song_path_dst, f'{STEM_DICT[i]}.wav')
                        sf.write(file_path_dst, data=segment_j.T, samplerate=sr, format='wav')


def split_test_and_valid(database_path, subset_size=20):
    """
    Split a random subset from the MUSDB18 test set.
    
    Args:
        database_path (str): The path to the MUSDB18 dataset.
        subset_size (int, optional): The desired size of the 'test' subset. Defaults to 20.
    
    Returns:
        subset (list): A subset of file names from MUSDB18.
    """

    database_test_path_src = os.path.join(database_path, 'test')
    database_test_list = os.listdir(database_test_path_src)
    subset = random.sample(database_test_list, subset_size)

    return subset

if __name__ == '__main__':

    database_path_src = r"data\musdb18"
    database_path_dst = r"data\musdb18_preprocessed"

    test_subset = split_test_and_valid(database_path_src)
    preprocess_musdb(database_path_src, database_path_dst, test_subset)