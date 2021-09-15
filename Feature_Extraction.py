import json
import os
import math
import librosa
import numpy as np

DATASET_PATH = "genres"
JSON_PATH = "extracted_data_bpm.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
NUM_SEGMENTS = 10


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):

    # dictionary to save the extracted features
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
        "tempo": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # extract tempo
                    onset_env = librosa.onset.onset_strength(signal, sr=sample_rate)
                    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)
                    tempo = np.rint(tempo)

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        data["tempo"].append(tempo.tolist())
                        print("{}, segment:{}".format(file_path.split("\\")[-1], d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        print('Writing the data in the .json file. This might take several minutes!')
        json.dump(data, fp, indent=4)


if __name__ == "__main__":

    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=NUM_SEGMENTS)
