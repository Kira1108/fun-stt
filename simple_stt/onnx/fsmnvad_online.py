from dataclasses import dataclass
from funasr_onnx import Fsmn_vad_online
from functools import lru_cache

# @lru_cache(maxsize=None)
def load_model():
    model_dir = "/Users/wanghuan/.cache/modelscope/hub/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    return Fsmn_vad_online(model_dir, disable_update=True)

def get_start_end(segments_result):
    speech_start = -1
    speech_end = -1
    
    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


# def get_start_end(segments_result):
#     speech_start = -1
#     speech_end = -1
    
#     if len(segments_result) == 0:
#         return speech_start, speech_end    
        
#     if segments_result[0][0] != -1:
#         speech_start = segments_result[0][0]
#     if segments_result[-1][-1] != -1:
#         speech_end = segments_result[-1][-1]
#     return speech_start, speech_end

@dataclass
class FsmnVadOnlineOnnx:

    def __post_init__(self):
        self.model = load_model()
        self.param_dict = {"in_cache": []}

    def clear_model_cache(self):
        self.param_dict['in_cache'] = []
        return self

    def run(self, speech, is_final: bool = False):
        
        if speech is None or len(speech) == 0:
            return -1, -1
        
        self.param_dict["is_final"] = is_final
        segments_result = self.model(audio_in=speech, param_dict=self.param_dict)
        # if segments_result is None or len(segments_result) == 0:
        #     return None
        
        if segments_result is None or len(segments_result) == 0:
            segments_result = [[]]
        
        start, end = get_start_end(segments_result[0])
        
        return start, end
    
    
if __name__ == "__main__":
    pass