import faulthandler
faulthandler.enable()
import sys
import os
os.environ["RKLLM_LOG_LEVEL"] = "1"
import ctypes
import argparse
import cv2
import numpy as np
import ztu_somemodelruntime_rknnlite2 as ort
from rkllm_binding import (
    RKLLMRuntime,
    RKLLMParam,
    RKLLMInput,
    RKLLMInferParam,
    LLMCallState,
    RKLLMInputType,
    RKLLMInferMode,
    RKLLMResult
)

# Constants
IMAGE_HEIGHT = 476
IMAGE_WIDTH = 476

def expand2square(img, background_color):
    """
    Expand the image into a square and fill it with the specified background color.
    """
    height, width, _ = img.shape
    if width == height:
        return img.copy()

    size = max(width, height)
    square_img = np.full((size, size, 3), background_color, dtype=np.uint8)

    x_offset = (size - width) // 2
    y_offset = (size - height) // 2

    square_img[y_offset:y_offset+height, x_offset:x_offset+width] = img
    return square_img

def llm_callback(result_ptr, userdata_ptr, state_enum):
    """
    Callback function to handle LLM results.
    """
    state = LLMCallState(state_enum)
    result = result_ptr.contents

    if state == LLMCallState.RKLLM_RUN_NORMAL:
        if result.text:
            print(result.text.decode('utf-8', errors='ignore'), end='', flush=True)
    elif state == LLMCallState.RKLLM_RUN_FINISH:
        print("\n", flush=True)
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        print("\nrun error", flush=True)
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Run RKLLM visual language model inference based on the C++ example."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("encoder_model_path", type=str, help="Path to the ONNX vision encoder model.")
    parser.add_argument("llm_model_path", type=str, help="Path to the .rkllm language model.")
    parser.add_argument("max_new_tokens", type=int, help="Maximum number of new tokens to generate.")
    parser.add_argument("max_context_len", type=int, help="Maximum context length.")
    # The rknn_core_num is not directly used by onnxruntime in the same way,
    # but we keep it for API consistency with the C++ example.
    # ONNX Runtime will manage its own threading and execution providers.
    parser.add_argument("rknn_core_num", type=int, help="Sets the number of npu cores used in vision encoder.")

    args = parser.parse_args()

    # --- 1. Initialize Image Encoder (ONNX Runtime) ---
    print("Initializing ONNX Runtime for vision encoder...")
    try:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = args.rknn_core_num
        ort_session = ort.InferenceSession(args.encoder_model_path, sess_options=sess_options)
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        sys.exit(1)
    print("Vision encoder loaded successfully.")
    
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    print(f"ONNX Input: {input_name}, ONNX Output: {output_name}")

    # --- 2. Initialize LLM ---
    print("Initializing RKLLM Runtime...")
    rk_llm = RKLLMRuntime()
    param = rk_llm.create_default_param()

    param.model_path = args.llm_model_path.encode('utf-8')
    param.top_k = 1
    param.max_new_tokens = args.max_new_tokens
    param.max_context_len = args.max_context_len
    param.skip_special_token = True
    param.img_start = b"<|vision_start|>"
    param.img_end = b"<|vision_end|>"
    param.img_content = b"<|image_pad|>"
    param.extend_param.base_domain_id = 1

    try:
        rk_llm.init(param, llm_callback)
        print("RKLLM initialized successfully.")
    except RuntimeError as e:
        print(f"RKLLM init failed: {e}")
        sys.exit(1)

    # --- 3. Image Preprocessing ---
    print("Preprocessing image...")
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Failed to read image from {args.image_path}")
        sys.exit(1)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    background_color = (127.5, 127.5, 127.5) # As per C++ example
    square_img = expand2square(img, background_color)
    resized_img = cv2.resize(square_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # Normalize and prepare for ONNX model
    input_tensor = resized_img.astype(np.float32)
    # Normalize using preprocessor config values
    input_tensor = (input_tensor / 255.0 - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
    # Convert to NCHW format
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension -> (1, 3, 392, 392)

    # --- 4. Run Image Encoder ---
    print("Running vision encoder...")
    import time
    start_time = time.time()
    try:
        img_vec_output = ort_session.run([output_name], {input_name: input_tensor.astype(np.float32)})[0]
        elapsed_time = time.time() - start_time
        print(f"视觉编码器推理耗时: {elapsed_time:.4f} 秒")
        # The output from C++ is a flat float array. Let's flatten the ONNX output.
        img_vec = img_vec_output.flatten().astype(np.float32)

    except Exception as e:
        print(f"Failed to run vision encoder inference: {e}")
        rk_llm.destroy()
        sys.exit(1)
    
    print("Image encoded successfully.")

    # --- 5. Interactive Chat Loop ---
    rkllm_infer_params = RKLLMInferParam()
    rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
    rkllm_infer_params.keep_history = 0

    # Set chat template
    rk_llm.set_chat_template(
        system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        prompt_prefix="<|im_start|>user\n",
        prompt_postfix="<|im_end|>\n<|im_start|>assistant\n"
    )

    pre_input = [
        "Picture 1: <image> What is in the image?",
        "Picture 1: <image> 这张图片中有什么？"
    ]
    print("\n**********************可输入以下问题对应序号获取回答/或自定义输入********************\n")
    for i, p in enumerate(pre_input):
        print(f"[{i}] {p}")
    print("\n*************************************************************************\n")

    try:
        while True:
            print("\nuser: ", end="", flush=True)
            input_str = sys.stdin.readline().strip()

            if not input_str:
                continue
            if input_str == "exit":
                break
            if input_str == "clear":
                try:
                    rk_llm.clear_kv_cache(keep_system_prompt=True)
                    print("KV cache cleared.")
                except RuntimeError as e:
                    print(f"Failed to clear KV cache: {e}")
                continue

            try:
                idx = int(input_str)
                if 0 <= idx < len(pre_input):
                    input_str = pre_input[idx]
                    print(input_str)
            except (ValueError, IndexError):
                pass # Use the raw string if not a valid index

            rkllm_input = RKLLMInput()
            rkllm_input.role = b"user"
            
            print("robot: ", end="", flush=True)

            if "<image>" in input_str:
                rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_MULTIMODAL
                
                # Setup multimodal input
                rkllm_input.multimodal_input.prompt = input_str.encode('utf-8')
                rkllm_input.multimodal_input.image_embed = img_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                rkllm_input.multimodal_input.n_image_tokens = img_vec_output.shape[0]
                print("n_image_tokens: ", rkllm_input.multimodal_input.n_image_tokens)
                rkllm_input.multimodal_input.n_image = 1
                rkllm_input.multimodal_input.image_height = IMAGE_HEIGHT
                rkllm_input.multimodal_input.image_width = IMAGE_WIDTH
            else:
                rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
                rkllm_input.prompt_input = input_str.encode('utf-8')

            try:
                rk_llm.run(rkllm_input, rkllm_infer_params)
            except RuntimeError as e:
                print(f"\nError during rkllm_run: {e}")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        print("Releasing resources...")
        rk_llm.destroy()
        print("RKLLM instance destroyed.")

if __name__ == "__main__":
    main()

