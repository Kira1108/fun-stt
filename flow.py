from simple_stt.utils import iterate_600ms_chunks, iterate_60ms_chunks
from simple_stt.onnx import FsmnVadOnlineOnnx
from simple_stt.onnx import ParaformerOnlineOnnx
from simple_stt.onnx import SeacoParaformerOfflineOnnx
from simple_stt.onnx import CTPuncOfflineOnnx
from simple_stt.onnx import ParaformerOfflineOnnx
import json
import numpy as np
import time

def process_wav(wav_path:str):
    data = []
    do_streaming = False
    do_offline = False
    frames = []
    frames_asr = []
    frames_asr_online = []
    is_speaking = True
    chunk_interval = 10
    vad_pre_idx = 0
    speech_start = False
    speech_end_i = -1
    mode = '2pass'

    online_asr = ParaformerOnlineOnnx()
    vad = FsmnVadOnlineOnnx()
    # offline_asr = SeacoParaformerOfflineOnnx()
    offline_asr = ParaformerOfflineOnnx()
    punc  = CTPuncOfflineOnnx()
    vad.clear_model_cache()
    online_asr.clear_model_cache()


    for chunk, final in iterate_60ms_chunks(wav_path):
        start = time.time()
        partial_text = None
        stable_text = None
        is_speaking = not final
        
        frames.append(chunk)
        frames_asr_online.append(chunk)
        duration_ms = len(chunk) // 16
        vad_pre_idx += duration_ms
        
        is_final = (speech_end_i != -1) or final
        
        if (len(frames_asr_online) % chunk_interval == 0) or is_final:
            if len(frames_asr_online) == 0:
                partial_text = ""
            else:
                audio_in = np.concatenate(frames_asr_online, axis=0)
                partial_text = online_asr.run(audio_in, is_final=is_final)
            frames_asr_online = []

        if speech_start:
            frames_asr.append(chunk)
            
        speech_start_i, speech_end_i = vad.run(chunk, is_final=False)
        
        if speech_start_i != -1:
            speech_start = True
            beg_bias = (vad_pre_idx - speech_start_i) // duration_ms
            frames_pre = frames[-beg_bias:]
            frames_asr = []
            frames_asr.extend(frames_pre)
            
        if speech_end_i != -1 or not is_speaking:
            print("Doing Offline ASR...")
            if len(frames_asr) == 0:
                print("Found empty frames.")
                stable_text = ""
            else:
                audio_in= np.concatenate(frames_asr, axis=0)
                stable_text = offline_asr.run(audio_in)
                print("Stable Text", stable_text)
                
            if len(stable_text) > 0:
                stable_text = punc.run(stable_text)
                data.append({'type':"stable",'text':stable_text, 'start': speech_start_i, 'end': speech_end_i})
            frames_asr = []
            frames_asr_online = []
            speech_start = False
            online_asr.clear_model_cache()
            
            if not is_speaking:
                vad_pre_idx = 0
                frames = []
                vad.clear_model_cache()
            else:
                frames = frames[-20:]
        end = time.time()
        
        print(f"Taken:{end-start} seconds, Start: {speech_start_i}, End: {speech_end_i}, Final: {final}, Partial: {partial_text}, Stable: {stable_text}")
    return data

if __name__ == "__main__": 
    
    # res = process_wav("./aishell/BAC009S0901W0350.wav")
    # print(res)
    
    # res = process_wav("./data-16k/log.20250305_audio_21.wav")
    # print(res)
    
    # run path   
    from pathlib import Path
    import json
    
    nfiles = 200
    current = 0
    for fp in Path("./aishell").glob("*.wav"):
        print(f"Processing {fp.name} ...")
        data = process_wav(str(fp))
        with open(Path("outputs") / f"{fp.stem}.json","w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        current += 1
        print(f"Processed {fp.name}, total {current} files.")
        if current >= nfiles:
            break