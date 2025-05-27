from dataclasses import dataclass
from funasr_onnx import CT_Transformer_VadRealtime
from functools import lru_cache
@lru_cache(maxsize=None)
def load_model():
    model_dir = "damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
    return CT_Transformer_VadRealtime(model_dir, disable_update=True)


@dataclass
class CTPuncOnlineOnnx:
    """这个模型本身是没有is_final这个参数的，所以直接传一个text就ok"""
    def __post_init__(self):
        self.model_dir =  "damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
        self.model = load_model()
        self.param_dict = {'cache': []}
        self.rec_result_all = ''

    def clear_model_cache(self):
        self.param_dict['cache'] = []
        return self

    def run(self, text:str) -> str:
        text_result = self.model(text, param_dict=self.param_dict)
        self.rec_result_all += text_result[0]
        return text_result[0]