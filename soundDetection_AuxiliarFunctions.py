# --------------- createAudiosDataset ---------------
#
# Method to read a csv file and create a dataframe from it.
# Parse the .csv file contents and generate fields pointing 
# to the audio paths
#
# metadataFile: Path to .csv file
# audiosPath: Path where the audios referred in the metadataFile are stored;
def createAudiosDataset(metadataFile, audiosPath):
    
    import pandas as pd
    import os

    #Create a dataframe for the metadata (.csv) table
    metadata_DF = pd.read_csv(metadataFile)

    # Create audio DataFrame + define columns
    audios_DF = pd.DataFrame({
    "path": [],
    "filename": [],
    "fold":[],
    "duration":[],
    "class":[],
    "classID":[]
    });

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
            dfRow['classID']]

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

    import librosa
    
    # Extract audio
    audioMatrix, samplingRate = librosa.load(filePath, sr=arg_sr, mono=True)
    return audioMatrix


# --------------- extractAudioSubset ---------------
#
# Method for extracting the audio subsets
# audio: Audio data to be segmented
# isEngine: 1 for engine, 0 for others
# audioOverlap: value from 0 to 1 indicating the segmenting overlap
# audioSegSize: Number of samples desired per segment
def extractAudioSubset(audio, isEngine, audioOverlap, audioSegSize):
    import numpy as np

    segAudio_A = []
    isEngine_A = []
    
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
            isEngine_A.append(isEngine)
        
    #else:
        #print('-> numSubSegs: ZERO - too short!')
    
    return segAudio_A, isEngine_A



