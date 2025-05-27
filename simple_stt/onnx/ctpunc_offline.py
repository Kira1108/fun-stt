from funasr_onnx import CT_Transformer
from dataclasses import dataclass

from functools import lru_cache

@lru_cache(maxsize=None)
def load_model():
    model_dir = "/Users/wanghuan/.cache/modelscope/hub/iic/punc_ct-transformer_cn-en-common-vocab471067-large"
    return CT_Transformer(model_dir, disable_update=True)

@dataclass
class CTPuncOfflineOnnx:

    def __post_init__(self):
        self.model = load_model()
        
    def run(self, text:str):
        if text is None or text.strip() == "":
            return ""
        return self.model(text)[0]