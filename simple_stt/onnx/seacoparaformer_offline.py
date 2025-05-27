from dataclasses import dataclass
from funasr_onnx import SeacoParaformer
from functools import lru_cache

@lru_cache(maxsize=None)
def load_model():
    model_dir = "/Users/wanghuan/.cache/modelscope/hub/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    return SeacoParaformer(model_dir, batch_size=1, disable_update=True)

@dataclass
class SeacoParaformerOfflineOnnx:

    def __post_init__(self):
        self.hotwords = "易鑫 序禄"
        self.model = load_model()

    def run(self, speech):
        try:
            return self.model(speech, self.hotwords)[0]['preds'].replace(" ", "")
        except:
            return None