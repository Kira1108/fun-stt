from dataclasses import dataclass
from funasr_onnx import Paraformer
from functools import lru_cache

@lru_cache(maxsize=None)
def load_model():
    model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    # model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    return Paraformer(model_dir, batch_size=1, quantize=False, disable_update=True)

@dataclass
class ParaformerOfflineOnnx:

    def __post_init__(self):
        self.model = load_model()
    
    def run(self, speech) -> str:
        try:
            return self.model(speech)[0]['preds'][0]
        except Exception as e:
            print("Offline ASR Error..." * 3)
            print(e)
            return None
        
if __name__ == "__main__":
    paraformer_offline = ParaformerOfflineOnnx()
    # Example usage
    # audio_data = ...  # Load your audio data here
    # result = paraformer_offline.run(audio_data)
    # print(result)