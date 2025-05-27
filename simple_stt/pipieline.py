import asyncio
import json
import websockets
import time
import logging
import tracemalloc
import numpy as np
import argparse
import ssl
from simple_stt.onnx import FsmnVadOnlineOnnx

    
vad_onnx = FsmnVadOnlineOnnx()

async def async_vad(websocket, audio_in, is_final:bool = False):

    segments_result = vad_onnx.run(speech=audio_in, is_final=is_final)

    speech_start = -1
    speech_end = -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


