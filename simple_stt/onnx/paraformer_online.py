from dataclasses import dataclass
from funasr_onnx.paraformer_online_bin import Paraformer
from functools import lru_cache

@lru_cache(maxsize=None)
def load_model():
    model_dir = "/Users/wanghuan/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    return Paraformer(
        model_dir, batch_size=1, quantize=True, chunk_size=[5, 10, 5], intra_op_num_threads=4, disable_update=True
    )

@dataclass
class ParaformerOnlineOnnx:

    def __post_init__(self):
        self.chunk_size = [5, 10, 5]
        self.model = load_model()
        self.param_dict = {"cache": dict()}
        self.final_result = ""

    def clear_model_cache(self):
        self.param_dict['cache'] = dict()
        return self

    def run(self, chunk, is_final:bool = False) -> str:
        if chunk is None or len(chunk) == 0:
            return None
        try:
            self.param_dict['is_final'] = is_final
            rec_result = self.model(audio_in=chunk, param_dict=self.param_dict)
            if len(rec_result) > 0:
                self.final_result += rec_result[0]["preds"][0]
                return rec_result[0]['preds'][0]
            return None
        except Exception as e:
            print(f"Error during streaming asr inference: {e}")
            return None