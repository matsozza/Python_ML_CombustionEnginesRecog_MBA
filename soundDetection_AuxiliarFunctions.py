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
    "filename": [],
    "duration":[],
    "class":[]
    });

    # Process data
    n_files = 0
    for index, dfRow in metadata_DF.iterrows():
        # Get audio file path
        audioFilePath = audiosPath + '/' + dfRow['slice_file_name']
        
        # Provide feedback to the user
        if (n_files % 1000 == 0):
            print(audioFilePath, " ---> Index: ", index) #Verbose
        
        # Compile results or provide error feedback
        if not os.path.exists(audioFilePath):
            print(dfRow['slice_file_name'] + " (index=" + str(n_files) + ") does not exist!")
        else:
            audios_DF.loc[len(audios_DF.index)] = [dfRow['slice_file_name'], dfRow['end']-dfRow['start'], dfRow['class']] 
            n_files += 1

    print("Number of processed audio files: " + str(n_files))

    return audios_DF


# --------------- extractSpectogram ---------------
#
# Method for extracting the audio track + spectogram. 
# The audio intensity is normalized between 0 and 1
def extractSpectogram(filePath, arg_nfft=2048,arg_hoplen=512,arg_nmels=26, sr=44100):

    import librosa
    import pandas as pd
    import numpy as np

    SR_CONST=44100
    
    # Extract audio + normalize
    audioMatrix, samplingRate = librosa.load(filePath, sr=SR_CONST, mono=True)
    numSamples = len(audioMatrix)
    timeArray = np.linspace(0, (numSamples-1)/SR_CONST, numSamples)

    audioMatrix = audioMatrix / np.max(audioMatrix) #Normalize per amplitude
    #audioMatrix = audioMatrix / np.max(librosa.feature.rms(y=audioMatrix)) # Normalize per signal RMS

    # Extract Mel Spectogram
    spectMatrix = librosa.feature.melspectrogram(y=audioMatrix, sr=SR_CONST, n_fft=arg_nfft,hop_length=arg_hoplen, n_mels=arg_nmels)
    spectMatrix = spectMatrix #/ np.max(spectMatrix) #Normalize per amplitude

    return audioMatrix, spectMatrix 


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

    print('------------ START--------')
    print('-> audioSize' , audioSize)

    if audioSize >= (1 + (1-audioOverlap)) * audioSegSize:
        # Calculate num of subsegments to split current audio
        numSubSegs = np.floor((audioSize - audioSegSize)/(audioSegSize*(1-audioOverlap))) +1
        print('-> numSubSegs', int(numSubSegs))

        # Create the array of subsegments 'subAudio_AA'
        for idxSubAudio in range(int(numSubSegs)):
            # print('---> ', idxSubAudio)
            subStartIdx = int((idxSubAudio) * audioSegSize * (1- audioOverlap))
            subEndIdx = int(subStartIdx +  audioSegSize )

            segAudio_A.append(audio[subStartIdx : subEndIdx])
            isEngine_A.append(isEngine)
        
    else:
        print('-> numSubSegs: ZERO - too short!')
    
    return segAudio_A, isEngine_A



