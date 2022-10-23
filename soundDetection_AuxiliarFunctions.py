
# Libraries
import pandas as pd
import os
import librosa
import numpy as np
import pandas as pd

# Constants and globals
dfCols = {
    "path": [],
    "filename": [],
    "fold":[],
    "duration":[],
    "class":[],
    "classID":[],
    "isEngine": []
    }

# --------------- createAudiosDataset ---------------
#
# Method to read a csv file and create a dataframe from it.
# Parse the .csv file contents and generate fields pointing 
# to the audio paths
#
# metadataFile: Path to .csv file
# audiosPath: Path where the audios referred in the metadataFile are stored;
def createAudiosDataset(metadataFile, audiosPath):
    #Create a dataframe for the metadata (.csv) table
    metadata_DF = pd.read_csv(metadataFile)

    # Create audio DataFrame + define columns
    audios_DF = pd.DataFrame(dfCols);

    # Process data originally from 'UrbanSounds8k' dataset and metadata file
    n_files = 0
    for index, dfRow in metadata_DF.iterrows():
        # Get audio file path
        audioFilePath = audiosPath + 'fold' + str(dfRow['fold']) + '/' + dfRow['slice_file_name']
        
        # Provide feedback to the user
        if (n_files % 1000 == 0):
            print(audioFilePath, " ---> Index: ", index) #Verbose
        
        # Compile results or provide error feedback
        if not os.path.exists(audioFilePath):
            print(dfRow['slice_file_name'] + " (index=" + str(n_files) + ") does not exist!")
        else:
            dataToConcat = [audioFilePath,
            dfRow['slice_file_name'], 
            dfRow['fold'], 
            (dfRow['end']-dfRow['start']),
            dfRow['class'], 
            dfRow['classID'],
            0]

            audios_DF.loc[len(audios_DF.index)] = dataToConcat
            n_files += 1

    print("Number of processed audio files: " + str(n_files))

    # Create the 'isEngine' column
    audios_DF['isEngine'] = audios_DF['class'].map({'engine_idling': 1, 'air_conditioner':0, 'car_horn':0,
                                       'children_playing':0, 'dog_bark':0, 'drilling':0, 'gun_shot':0,
                                       'jackhammer':0, 'siren':0, 'street_music':0}, na_action=None)

    return audios_DF


# --------------- loadAudioTrack ---------------
#
# Method for extracting the audio track data (array).
def loadAudioTrack(filePath, arg_sr=44100):
    # Extract audio
    audioMatrix, samplingRate = librosa.load(filePath, sr=arg_sr, mono=True)
    return audioMatrix


# --------------- extractAudioSubset ---------------
#
# Method for extracting the audio subsets
# audio: Audio data to be segmented
# dataFrame: Receive dataframe row corresponding to the audio to be segmented
# audioOverlap: value from 0 to 1 indicating the segmenting overlap
# audioSegSize: Number of samples desired per segment
def extractAudioSubset(audio, dataFrame, audioOverlap, audioSegSize, sr):
    segAudio_A = []
    segAudio_DF = pd.DataFrame(dfCols);
    
    # Get audio length / total duration
    audioSize = (audio.shape[0])

    #print('------------ START--------')
    #print('-> audioSize' , audioSize)

    if audioSize >= (1 + (1-audioOverlap)) * audioSegSize:
        # Calculate num of subsegments to split current audio
        numSubSegs = np.floor((audioSize - audioSegSize)/(audioSegSize*(1-audioOverlap))) +1
        #print('-> numSubSegs', int(numSubSegs))

        # Create the array of subsegments 'subAudio_AA'
        for idxSubAudio in range(int(numSubSegs)):
            # print('---> ', idxSubAudio)
            subStartIdx = int((idxSubAudio) * audioSegSize * (1- audioOverlap))
            subEndIdx = int(subStartIdx +  audioSegSize )

            segAudio_A.append(audio[subStartIdx : subEndIdx])

            segAudio_DF.loc[len(segAudio_DF.index)] = dataFrame
            segAudio_DF['duration'] = audioSegSize/sr # Fix duration for the new snippet
    #else:
        #print('-> numSubSegs: ZERO - too short!')
    
    return segAudio_A, segAudio_DF


# --------------- trainTestFolder ---------------
#
# Custom method to pick train/test dataframes based on folders
# testFolders: Folders to be included in the 'test' dataframe (array)
# audio_DF: Dataframe containing the sound mappings
def trainTestFolder(testFolders, audio_DF, audio_AA):
    train_DF = pd.DataFrame(dfCols)
    test_DF = pd.DataFrame(dfCols)
    train_AA = []
    test_AA = []
    
    for idx, dfRow in audio_DF.iterrows():
        if dfRow['fold'] in testFolders:
            test_DF.loc[len(test_DF.index)] = dfRow
            test_AA.append(audio_AA[idx])
        else:
            train_DF.loc[len(train_DF.index)] = dfRow
            train_AA.append(audio_AA[idx])

    return train_DF, test_DF, train_AA, test_AA
