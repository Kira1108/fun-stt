import numpy as np
import soundfile

def iterate_600ms_chunks(wav_path:str):

    """
    for chunk in iterate_600ms_chunks("/content/BAC009S0901W0126.wav"):
        print(len(chunk))
    """
    speech, sample_rate = soundfile.read(wav_path)
    speech_length = speech.shape[0]
    sample_offset = 0
    step = 9600

    for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):

        if sample_offset + step >= speech_length - 1:
            step = speech_length - sample_offset
            is_final = True
        else:
            is_final = False

        yield speech[sample_offset : sample_offset + step], is_final
        

def iterate_60ms_chunks(wav_path:str):

    """
    for chunk in iterate_600ms_chunks("/content/BAC009S0901W0126.wav"):
        print(len(chunk))
    """
    speech, sample_rate = soundfile.read(wav_path)
    speech_length = speech.shape[0]
    sample_offset = 0
    step = 960

    for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):

        if sample_offset + step >= speech_length - 1:
            step = speech_length - sample_offset
            is_final = True
        else:
            is_final = False

        yield speech[sample_offset : sample_offset + step], is_final