#!/usr/bin/env python
# coding: utf-8

import datetime
import argparse
from rknn.api import RKNN
from sys import exit


parser = argparse.ArgumentParser(description='Convert ONNX to RKNN model.')
parser.add_argument('onnx_model', type=str, help='Path to the input ONNX model file.')
args = parser.parse_args()


ONNX_MODEL = args.onnx_model
RKNN_MODEL = ONNX_MODEL.replace(".onnx", ".rknn")
DATASET = "/home/zt/rk3588-nn/rknn_model_zoo/datasets/COCO/coco_subset_20.txt"
QUANTIZE = False
detailed_performance_log = True

timedate_iso = datetime.datetime.now().isoformat()

rknn = RKNN(verbose=True)
rknn.config(
    # mean_values=[x * 255 for x in [0.485, 0.456, 0.406]],
    # std_values=[x * 255 for x in [0.229, 0.224, 0.225]],
    quantized_dtype="w8a8",
    quantized_algorithm="normal",
    quantized_method="channel",
    quantized_hybrid_level=0,
    target_platform="rk3588",
    quant_img_RGB2BGR=False,
    float_dtype="float16",
    optimization_level=3,
    custom_string=f"converted by: email: 2302004040@qq.com at {timedate_iso}",
    remove_weight=False,
    compress_weight=False,
    inputs_yuv_fmt=None,
    single_core_mode=False,
    # dynamic_input=[  #这个和下面的inputs + input_size_list二选一
    #     [
    #         [1, 3, 240, 320],
    #         # ...
    #     ],
    #     [
    #         [1, 3, 480, 640],
    #         # ...
    #     ],
    #     [
    #         [1, 3, 960, 1280],
    #         # ...
    #     ],
    # ],
    model_pruning=False,
    op_target={'Gather':'cpu'},
    quantize_weight=False,
    remove_reshape=False,
    sparse_infer=False,
    enable_flash_attention=False,
    # 隐藏的参数
    # disable_rules=[],
    # sram_prefer=False,
    # nbuf_prefer=False,
    # check_data=[],
)

ret = rknn.load_onnx(model=ONNX_MODEL)
ret = rknn.build(do_quantization=QUANTIZE, dataset=DATASET, rknn_batch_size=None)
ret = rknn.export_rknn(RKNN_MODEL)

# ret = rknn.init_runtime(target='rk3588',core_mask=RKNN.NPU_CORE_0,perf_debug=detailed_performance_log)
# rknn.eval_perf()
# ret = rknn.accuracy_analysis(inputs=['processed_images_rknn.npy'], target='rk3588')
