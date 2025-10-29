⚠️ This repository is synchronized from Hugging Face repository `happyme531/Qwen2.5-VL-3B-Instruct-RKLLM`.

---
base_model:
- Qwen/Qwen2.5-VL-3B-Instruct
tags:
- rknn
- rkllm
---
# Qwen2.5-VL-3B-Instruct-RKLLM

## (English README see below)

在RK3588上运行强大的Qwen2.5-VL-3B-Instruct-RKLLM视觉大模型!

- 推理速度(RK3588): 视觉编码器 3.4s(三核并行) + LLM 填充 2.3s (320 tokens / 138 tps) + 解码 8.2 tps
- 内存占用(RK3588, 上下文长度1024): 6.1GB

## 使用方法

1. 克隆或者下载此仓库到本地. 模型较大, 请确保有足够的磁盘空间.
   
2. 开发板的RKNPU2内核驱动版本必须>=0.9.6才能运行这么大的模型. 
   使用root权限运行以下命令检查驱动版本:
   ```bash
   > cat /sys/kernel/debug/rknpu/version 
   RKNPU driver: v0.9.8
   ```
   如果版本过低, 请更新驱动. 你可能需要更新内核, 或查找官方文档以获取帮助.
   
3. 安装依赖

```bash
pip install "numpy<2" opencv-python rknn-toolkit-lite2
```

4. 运行
   
```bash
python ./run_rkllm.py ./test.jpg ./vision_encoder.rknn ./language_model_w8a8.rkllm 512 1024 3
```

参数说明:
- `512`: max_new_tokens, 最大生成token数.
- `1024`: max_context_len, 最大上下文长度.
- `3`: npu_core_num, 使用的NPU核心数.

如果实测性能不理想, 可以调整CPU调度器让CPU始终运行在最高频率, 并把推理程序绑定到大核(`taskset -c 4-7 python ...`)

test.jpg:
![test.jpg](./test.jpg)

```
Initializing ONNX Runtime for vision encoder...
W rknn-toolkit-lite2 version: 2.3.2
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning message.)
Vision encoder loaded successfully.
ONNX Input: pixel_values, ONNX Output: vision_features
Initializing RKLLM Runtime...
I rkllm: rkllm-runtime version: 1.2.1, rknpu driver version: 0.9.8, platform: RK3588
I rkllm: loading rkllm model from ./language_model_w8a8.rkllm
I rkllm: rkllm-toolkit version: 1.2.1, max_context_limit: 4096, npu_core_num: 3, target_platform: RK3588, model_dtype: W8A8
I rkllm: Enabled cpus: [4, 5, 6, 7]
I rkllm: Enabled cpus num: 4
I rkllm: Using mrope
RKLLM initialized successfully.
Preprocessing image...
Running vision encoder...
W The input[0] need NHWC data format, but NCHW set, the data format and data buffer will be changed to NHWC.
视觉编码器推理耗时: 3.5427 秒
Image encoded successfully.
I rkllm: reset chat template:
I rkllm: system_prompt: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
I rkllm: prompt_prefix: <|im_start|>user\n
I rkllm: prompt_postfix: <|im_end|>\n<|im_start|>assistant\n
W rkllm: Calling rkllm_set_chat_template will disable the internal automatic chat template parsing, including enable_thinking. Make sure your custom prompt is complete and valid.

**********************可输入以下问题对应序号获取回答/或自定义输入********************

[0] Picture 1: <image> What is in the image?
[1] Picture 1: <image> 这张图片中有什么？

*************************************************************************


user: 0
Picture 1: <image> What is in the image?
robot: n_image_tokens:  289
The image shows a cozy bedroom with several notable features:

- A large bed covered with a blue comforter.
- A wooden dresser next to the bed, topped with various items including a mirror and some decorative objects.
- A window allowing natural light into the room, offering a view of greenery outside.
- A bookshelf filled with numerous books on shelves.
- A basket placed near the foot of the bed.
- A lamp on a side table beside the bed.

The overall ambiance is warm and inviting.

I rkllm: --------------------------------------------------------------------------------------
I rkllm:  Model init time (ms)  3361.48                                                                    
I rkllm: --------------------------------------------------------------------------------------
I rkllm:  Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second      
I rkllm: --------------------------------------------------------------------------------------
I rkllm:  Prefill       2201.45          321       6.86                     145.81                 
I rkllm:  Generate      12419.47         102       121.76                   8.21                   
I rkllm: --------------------------------------------------------------------------------------
I rkllm:  Peak Memory Usage (GB)
I rkllm:  6.19        
I rkllm: --------------------------------------------------------------------------------------

user: 1
Picture 1: <image> 这张图片中有什么？
robot: n_image_tokens:  289
这张照片展示了一个卧室的内部。房间有一扇大窗户，可以看到外面的绿色植物。房间里有各种物品：一个蓝色的大床单覆盖在一张床上；一盏灯放在梳妆台上；一面镜子挂在墙上；书架上摆满了书籍和一些装饰品；还有一些篮子、花盆和其他小物件散落在周围。

I rkllm: --------------------------------------------------------------------------------------
I rkllm:  Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second      
I rkllm: --------------------------------------------------------------------------------------
I rkllm:  Prefill       184.35           13        14.18                    70.52                  
I rkllm:  Generate      8711.49          72        120.99                   8.26                   
I rkllm: --------------------------------------------------------------------------------------
I rkllm:  Peak Memory Usage (GB)
I rkllm:  6.19        
I rkllm: --------------------------------------------------------------------------------------
```

## 模型转换

#### 准备工作

1. 安装rknn-toolkit2以及rkllm-toolkit:
```bash
pip install -U rknn-toolkit2 
```
rkllm-toolkit需要在这里手动下载: https://github.com/airockchip/rknn-llm/tree/main/rkllm-toolkit

2. 下载此仓库到本地, 但不需要下载`.rkllm`和`.rknn`结尾的模型文件.
3. 下载Qwen2.5-VL-3B-Instruct的huggingface模型仓库到本地. ( https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct )
  
#### 转换LLM

将`rkllm-convert.py`拷贝到Qwen2.5-VL-3B-Instruct的模型文件夹中，执行:
```bash
python rkllm-convert.py
```
默认是w8a8量化的，你可以自行打开脚本修改量化方式等。

#### 转换视觉编码器

1. 导出ONNX

将`export_vision_onnx.py`拷贝到Qwen2.5-VL-3B-Instruct的模型文件夹根目录中，然后**在该根目录**下执行:
```bash
mkdir vision
python ./export_vision_onnx.py . --savepath ./vision/vision_encoder.onnx
```
视觉编码器会导出到`vision/vision_encoder.onnx`. 默认宽高为476，你可以自行通过`--height`和`--width`参数修改。

2. 模型优化 (可选)

从 https://github.com/happyme531/rknn-toolkit2-utils 下载`split_matmul_onnx_profile.py`, 之后运行:
```bash
python ./split_matmul_onnx_profile.py --input vision/vision_encoder.onnx --output vision_encoder_opt.onnx  --pattern "/visual/blocks\..*?/mlp/down_proj.*" --factor 5  
```
优化后的模型会输出到`vision_encoder_opt.onnx`

3. 转换rknn

```bash
python ./convert_vision_encoder.py ./vision_encoder_opt.onnx
```
(这一步可能需要20分钟以上)
转换后模型会输出到`vision_encoder_opt.rknn`

为了与"使用方法"中的命令保持一致, 你可以将其重命名:
```bash
mv vision_encoder_opt.rknn vision_encoder.rknn
```

## 已知问题

- 由于RKLLM的多模态输入的限制, 在整个对话中只能加载一张图片.
- 没有实现多轮对话.
- RKLLM的w8a8量化貌似存在不小的精度损失.
- 可能由于RKNPU2的访存模式问题，输入尺寸边长不为64的整数倍时模型运行速度会有奇怪的明显提升。

## 参考

- [Qwen/Qwen2.5-VL-3B-Instruct-RKLLM](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-RKLLM)

---

# English README

Run the powerful Qwen2.5-VL-3B-Instruct-RKLLM vision large model on RK3588!

- **Inference Speed (RK3588)**: Vision Encoder 3.4s (3-core parallel) + LLM Prefill 2.3s (320 tokens / 138 tps) + Decode 8.2 tps
- **Memory Usage (RK3588, context length 1024)**: 6.1GB

## How to Use

1.  Clone or download this repository locally. The model is large, so ensure you have enough disk space.

2.  The RKNPU2 kernel driver version on your board must be `>=0.9.6` to run such a large model. Run the following command with root privileges to check the driver version:
    ```bash
    > cat /sys/kernel/debug/rknpu/version
    RKNPU driver: v0.9.8
    ```
    If the version is too old, please update the driver. You may need to update your kernel or consult the official documentation for help.

3.  Install dependencies:
    ```bash
    pip install "numpy<2" opencv-python rknn-toolkit-lite2
    ```

4.  Run the model:
    ```bash
    python ./run_rkllm.py ./test.jpg ./vision_encoder.rknn ./language_model_w8a8.rkllm 512 1024 3
    ```
    **Parameter Descriptions:**
    - `512`: `max_new_tokens`, the maximum number of tokens to generate.
    - `1024`: `max_context_len`, the maximum context length.
    - `3`: `npu_core_num`, the number of NPU cores to use.

If the performance is not ideal, you can adjust the CPU scheduler to keep the CPU running at its highest frequency and bind the inference program to the big cores (`taskset -c 4-7 python ...`).

The example output is shown in the Chinese section above.

## Model Conversion

#### Prerequisites

1.  Install rknn-toolkit2 and rkllm-toolkit:
    ```bash
    pip install -U rknn-toolkit2
    ```
    rkllm-toolkit needs to be downloaded manually from here: https://github.com/airockchip/rknn-llm/tree/main/rkllm-toolkit

2.  Download this repository locally, but you don't need the model files ending with `.rkllm` and `.rknn`.
3.  Download the Qwen2.5-VL-3B-Instruct huggingface model repository locally from: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

#### Convert LLM

Copy `rkllm-convert.py` into the Qwen2.5-VL-3B-Instruct model folder and execute:
```bash
python rkllm-convert.py
```
It uses w8a8 quantization by default. You can open the script to modify the quantization method and other settings.

#### Convert Vision Encoder

1.  **Export ONNX**

    Copy `export_vision_onnx.py` to the root directory of the Qwen2.5-VL-3B-Instruct model folder, then execute the following **in the root directory**:
    ```bash
    mkdir vision
    python ./export_vision_onnx.py . --savepath ./vision/vision_encoder.onnx
    ```
    The vision encoder will be exported to `vision/vision_encoder.onnx`. The default height and width are 476, which you can modify using the `--height` and `--width` parameters.

2.  **Model Optimization (Optional)**

    Download `split_matmul_onnx_profile.py` from https://github.com/happyme531/rknn-toolkit2-utils, then run:
    ```bash
    python ./split_matmul_onnx_profile.py --input vision/vision_encoder.onnx --output vision_encoder_opt.onnx  --pattern "/visual/blocks\..*?/mlp/down_proj.*" --factor 5
    ```
    The optimized model will be saved as `vision_encoder_opt.onnx`.

3.  **Convert to RKNN**

    ```bash
    python ./convert_vision_encoder.py ./vision_encoder_opt.onnx
    ```
    (This step may take over 20 minutes)

    The converted model will be saved as `vision_encoder_opt.rknn`. To match the command in the "How to Use" section, you can rename it:
    ```bash
    mv vision_encoder_opt.rknn vision_encoder.rknn
    ```

## Known Issues

- Due to limitations in RKLLM's multimodal input, only one image can be loaded per conversation.
- Multi-turn conversation is not implemented.
- The w8a8 quantization in RKLLM seems to cause a non-trivial loss of precision.
- Possibly due to memory access patterns of the RKNPU2, weirdly the model runs faster when the input image dimensions are not multiples of 64.

## References

- [Qwen/Qwen2.5-VL-3B-Instruct-RKLLM](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-RKLLM)
