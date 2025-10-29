import ctypes
import enum
import os

# Define constants from the header
CPU0 = (1 << 0)  # 0x01
CPU1 = (1 << 1)  # 0x02
CPU2 = (1 << 2)  # 0x04
CPU3 = (1 << 3)  # 0x08
CPU4 = (1 << 4)  # 0x10
CPU5 = (1 << 5)  # 0x20
CPU6 = (1 << 6)  # 0x40
CPU7 = (1 << 7)  # 0x80

# --- Enums ---
class LLMCallState(enum.IntEnum):
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3

class RKLLMInputType(enum.IntEnum):
    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1
    RKLLM_INPUT_EMBED = 2
    RKLLM_INPUT_MULTIMODAL = 3

class RKLLMInferMode(enum.IntEnum):
    RKLLM_INFER_GENERATE = 0
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
    RKLLM_INFER_GET_LOGITS = 2

# --- Structures ---
class RKLLMExtendParam(ctypes.Structure):
    base_domain_id: ctypes.c_int32
    embed_flash: ctypes.c_int8
    enabled_cpus_num: ctypes.c_int8
    enabled_cpus_mask: ctypes.c_uint32
    n_batch: ctypes.c_uint8
    use_cross_attn: ctypes.c_int8
    reserved: ctypes.c_uint8 * 104

    _fields_ = [
        ("base_domain_id", ctypes.c_int32),     # 基础域ID
        ("embed_flash", ctypes.c_int8),         # 是否从闪存查询词嵌入向量（1启用，0禁用）
        ("enabled_cpus_num", ctypes.c_int8),    # 推理启用的CPU数量
        ("enabled_cpus_mask", ctypes.c_uint32), # 指示启用哪些CPU的位掩码
        ("n_batch", ctypes.c_uint8),            # 一次前向传播中并发处理的输入样本数，设置>1启用批量推理，默认为1
        ("use_cross_attn", ctypes.c_int8),      # 是否启用交叉注意力（非零启用，0禁用）
        ("reserved", ctypes.c_uint8 * 104)     # 保留字段
    ]

class RKLLMParam(ctypes.Structure):
    model_path: ctypes.c_char_p
    max_context_len: ctypes.c_int32
    max_new_tokens: ctypes.c_int32
    top_k: ctypes.c_int32
    n_keep: ctypes.c_int32
    top_p: ctypes.c_float
    temperature: ctypes.c_float
    repeat_penalty: ctypes.c_float
    frequency_penalty: ctypes.c_float
    presence_penalty: ctypes.c_float
    mirostat: ctypes.c_int32
    mirostat_tau: ctypes.c_float
    mirostat_eta: ctypes.c_float
    skip_special_token: ctypes.c_bool
    is_async: ctypes.c_bool
    img_start: ctypes.c_char_p
    img_end: ctypes.c_char_p
    img_content: ctypes.c_char_p
    extend_param: RKLLMExtendParam

    _fields_ = [
        ("model_path", ctypes.c_char_p),         # 模型文件路径
        ("max_context_len", ctypes.c_int32),     # 上下文窗口最大token数
        ("max_new_tokens", ctypes.c_int32),      # 最大生成新token数
        ("top_k", ctypes.c_int32),               # Top-K采样参数
        ("n_keep", ctypes.c_int32),              # 上下文窗口移动时保留的kv缓存数量
        ("top_p", ctypes.c_float),               # Top-P（nucleus）采样参数
        ("temperature", ctypes.c_float),         # 采样温度，影响token选择的随机性
        ("repeat_penalty", ctypes.c_float),      # 重复token惩罚
        ("frequency_penalty", ctypes.c_float),   # 频繁token惩罚
        ("presence_penalty", ctypes.c_float),    # 输入中已存在token的惩罚
        ("mirostat", ctypes.c_int32),            # Mirostat采样策略标志（0表示禁用）
        ("mirostat_tau", ctypes.c_float),        # Mirostat采样Tau参数
        ("mirostat_eta", ctypes.c_float),        # Mirostat采样Eta参数
        ("skip_special_token", ctypes.c_bool),   # 是否跳过特殊token
        ("is_async", ctypes.c_bool),             # 是否异步推理
        ("img_start", ctypes.c_char_p),          # 多模态输入中图像的起始位置
        ("img_end", ctypes.c_char_p),            # 多模态输入中图像的结束位置
        ("img_content", ctypes.c_char_p),        # 图像内容指针
        ("extend_param", RKLLMExtendParam)       # 扩展参数
    ]

class RKLLMLoraAdapter(ctypes.Structure):
    lora_adapter_path: ctypes.c_char_p
    lora_adapter_name: ctypes.c_char_p
    scale: ctypes.c_float

    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]

class RKLLMEmbedInput(ctypes.Structure):
    embed: ctypes.POINTER(ctypes.c_float)
    n_tokens: ctypes.c_size_t

    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMTokenInput(ctypes.Structure):
    input_ids: ctypes.POINTER(ctypes.c_int32)
    n_tokens: ctypes.c_size_t

    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMMultiModelInput(ctypes.Structure):
    prompt: ctypes.c_char_p
    image_embed: ctypes.POINTER(ctypes.c_float)
    n_image_tokens: ctypes.c_size_t
    n_image: ctypes.c_size_t
    image_width: ctypes.c_size_t
    image_height: ctypes.c_size_t

    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]

class RKLLMCrossAttnParam(ctypes.Structure):
    """
    交叉注意力参数结构体
    
    该结构体用于在解码器中执行交叉注意力时使用。
    它提供编码器输出（键/值缓存）、位置索引和注意力掩码。
    
    - encoder_k_cache必须存储在连续内存中，布局为：
      [num_layers][num_tokens][num_kv_heads][head_dim]
    - encoder_v_cache必须存储在连续内存中，布局为：
      [num_layers][num_kv_heads][head_dim][num_tokens]
    """
    encoder_k_cache: ctypes.POINTER(ctypes.c_float)
    encoder_v_cache: ctypes.POINTER(ctypes.c_float)
    encoder_mask: ctypes.POINTER(ctypes.c_float)
    encoder_pos: ctypes.POINTER(ctypes.c_int32)
    num_tokens: ctypes.c_int

    _fields_ = [
        ("encoder_k_cache", ctypes.POINTER(ctypes.c_float)),  # 编码器键缓存指针（大小：num_layers * num_tokens * num_kv_heads * head_dim）
        ("encoder_v_cache", ctypes.POINTER(ctypes.c_float)),  # 编码器值缓存指针（大小：num_layers * num_kv_heads * head_dim * num_tokens）
        ("encoder_mask", ctypes.POINTER(ctypes.c_float)),     # 编码器注意力掩码指针（大小：num_tokens的数组）
        ("encoder_pos", ctypes.POINTER(ctypes.c_int32)),      # 编码器token位置指针（大小：num_tokens的数组）
        ("num_tokens", ctypes.c_int)                          # 编码器序列中的token数量
    ]

class RKLLMPerfStat(ctypes.Structure):
    """
    性能统计结构体
    
    用于保存预填充和生成阶段的性能统计信息。
    """
    prefill_time_ms: ctypes.c_float
    prefill_tokens: ctypes.c_int
    generate_time_ms: ctypes.c_float
    generate_tokens: ctypes.c_int
    memory_usage_mb: ctypes.c_float

    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),   # 预填充阶段总耗时（毫秒）
        ("prefill_tokens", ctypes.c_int),      # 预填充阶段处理的token数量
        ("generate_time_ms", ctypes.c_float),  # 生成阶段总耗时（毫秒）
        ("generate_tokens", ctypes.c_int),     # 生成阶段处理的token数量
        ("memory_usage_mb", ctypes.c_float)    # 推理期间VmHWM常驻内存使用量（MB）
    ]

class _RKLLMInputUnion(ctypes.Union):
    prompt_input: ctypes.c_char_p
    embed_input: RKLLMEmbedInput
    token_input: RKLLMTokenInput
    multimodal_input: RKLLMMultiModelInput

    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput)
    ]

class RKLLMInput(ctypes.Structure):
    """
    LLM输入结构体
    
    通过联合体表示不同类型的LLM输入。
    """
    role: ctypes.c_char_p
    enable_thinking: ctypes.c_bool
    input_type: ctypes.c_int
    _union_data: _RKLLMInputUnion

    _fields_ = [
        ("role", ctypes.c_char_p),              # 消息角色："user"（用户输入）、"tool"（函数结果）
        ("enable_thinking", ctypes.c_bool),     # 控制Qwen3模型是否启用"思考模式"
        ("input_type", ctypes.c_int),           # 枚举类型，指定输入类型（如prompt、token、embed、multimodal）
        ("_union_data", _RKLLMInputUnion)       # 联合体数据
    ]
    # Properties to make accessing union members easier
    @property
    def prompt_input(self) -> bytes: # Assuming c_char_p maps to bytes
        if self.input_type == RKLLMInputType.RKLLM_INPUT_PROMPT:
            return self._union_data.prompt_input
        raise AttributeError("Not a prompt input")
    @prompt_input.setter
    def prompt_input(self, value: bytes): # Assuming c_char_p maps to bytes
        if self.input_type == RKLLMInputType.RKLLM_INPUT_PROMPT:
            self._union_data.prompt_input = value
        else:
            raise AttributeError("Not a prompt input")
    @property
    def embed_input(self) -> RKLLMEmbedInput:
        if self.input_type == RKLLMInputType.RKLLM_INPUT_EMBED:
            return self._union_data.embed_input
        raise AttributeError("Not an embed input")
    @embed_input.setter
    def embed_input(self, value: RKLLMEmbedInput):
        if self.input_type == RKLLMInputType.RKLLM_INPUT_EMBED:
            self._union_data.embed_input = value
        else:
            raise AttributeError("Not an embed input")

    @property
    def token_input(self) -> RKLLMTokenInput:
        if self.input_type == RKLLMInputType.RKLLM_INPUT_TOKEN:
            return self._union_data.token_input
        raise AttributeError("Not a token input")
    @token_input.setter
    def token_input(self, value: RKLLMTokenInput):
        if self.input_type == RKLLMInputType.RKLLM_INPUT_TOKEN:
            self._union_data.token_input = value
        else:
            raise AttributeError("Not a token input")

    @property
    def multimodal_input(self) -> RKLLMMultiModelInput:
        if self.input_type == RKLLMInputType.RKLLM_INPUT_MULTIMODAL:
            return self._union_data.multimodal_input
        raise AttributeError("Not a multimodal input")
    @multimodal_input.setter
    def multimodal_input(self, value: RKLLMMultiModelInput):
        if self.input_type == RKLLMInputType.RKLLM_INPUT_MULTIMODAL:
            self._union_data.multimodal_input = value
        else:
            raise AttributeError("Not a multimodal input")

class RKLLMLoraParam(ctypes.Structure): # For inference
    lora_adapter_name: ctypes.c_char_p

    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]

class RKLLMPromptCacheParam(ctypes.Structure): # For inference
    save_prompt_cache: ctypes.c_int # bool-like
    prompt_cache_path: ctypes.c_char_p

    _fields_ = [
        ("save_prompt_cache", ctypes.c_int), # bool-like
        ("prompt_cache_path", ctypes.c_char_p)
    ]

class RKLLMInferParam(ctypes.Structure):
    mode: ctypes.c_int
    lora_params: ctypes.POINTER(RKLLMLoraParam)
    prompt_cache_params: ctypes.POINTER(RKLLMPromptCacheParam)
    keep_history: ctypes.c_int # bool-like

    _fields_ = [
        ("mode", ctypes.c_int), # Enum will be passed as int, changed RKLLMInferMode to ctypes.c_int
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int) # bool-like
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    hidden_states: ctypes.POINTER(ctypes.c_float)
    embd_size: ctypes.c_int
    num_tokens: ctypes.c_int

    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResultLogits(ctypes.Structure):
    logits: ctypes.POINTER(ctypes.c_float)
    vocab_size: ctypes.c_int
    num_tokens: ctypes.c_int

    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResult(ctypes.Structure):
    """
    LLM推理结果结构体
    
    表示LLM推理的结果，包含生成的文本、token ID、隐藏层状态、logits和性能统计。
    """
    text: ctypes.c_char_p
    token_id: ctypes.c_int32
    last_hidden_layer: RKLLMResultLastHiddenLayer
    logits: RKLLMResultLogits
    perf: RKLLMPerfStat

    _fields_ = [
        ("text", ctypes.c_char_p),                                  # 生成的文本结果
        ("token_id", ctypes.c_int32),                              # 生成的token ID
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),         # 最后一层的隐藏状态（如果请求的话）
        ("logits", RKLLMResultLogits),                             # 模型输出的logits
        ("perf", RKLLMPerfStat)                                    # 性能统计（预填充和生成）
    ]

# --- Typedefs ---
LLMHandle = ctypes.c_void_p

# --- Callback Function Type ---
LLMResultCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,  # 返回类型：int，表示处理状态
    ctypes.POINTER(RKLLMResult),  # LLM结果指针
    ctypes.c_void_p,              # 用户数据指针
    ctypes.c_int                  # LLM调用状态（LLMCallState枚举值）
)
"""
回调函数类型定义

用于处理LLM结果的回调函数。

参数：
- result: 指向LLM结果的指针
- userdata: 回调的用户数据指针  
- state: LLM调用状态（例如：完成、错误）

返回值：
- 0: 正常继续推理
- 1: 暂停推理。如果用户想要修改或干预结果（例如编辑输出、注入新提示），
     返回1以暂停当前推理。稍后，使用更新的内容调用rkllm_run来恢复推理。
"""

class RKLLMRuntime:
    def __init__(self, library_path="./librkllmrt.so"):
        try:
            self.lib = ctypes.CDLL(library_path)
        except OSError as e:
            raise OSError(f"Failed to load RKLLM library from {library_path}. "
                          f"Ensure it's in your LD_LIBRARY_PATH or provide the full path. Error: {e}")
        self._setup_functions()
        self.llm_handle = LLMHandle()
        self._c_callback = None # To keep the callback object alive

    def _setup_functions(self):
        # RKLLMParam rkllm_createDefaultParam();
        self.lib.rkllm_createDefaultParam.restype = RKLLMParam
        self.lib.rkllm_createDefaultParam.argtypes = []

        # int rkllm_init(LLMHandle* handle, RKLLMParam* param, LLMResultCallback callback);
        self.lib.rkllm_init.restype = ctypes.c_int
        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(LLMHandle),
            ctypes.POINTER(RKLLMParam),
            LLMResultCallback
        ]

        # int rkllm_load_lora(LLMHandle handle, RKLLMLoraAdapter* lora_adapter);
        self.lib.rkllm_load_lora.restype = ctypes.c_int
        self.lib.rkllm_load_lora.argtypes = [LLMHandle, ctypes.POINTER(RKLLMLoraAdapter)]

        # int rkllm_load_prompt_cache(LLMHandle handle, const char* prompt_cache_path);
        self.lib.rkllm_load_prompt_cache.restype = ctypes.c_int
        self.lib.rkllm_load_prompt_cache.argtypes = [LLMHandle, ctypes.c_char_p]

        # int rkllm_release_prompt_cache(LLMHandle handle);
        self.lib.rkllm_release_prompt_cache.restype = ctypes.c_int
        self.lib.rkllm_release_prompt_cache.argtypes = [LLMHandle]

        # int rkllm_destroy(LLMHandle handle);
        self.lib.rkllm_destroy.restype = ctypes.c_int
        self.lib.rkllm_destroy.argtypes = [LLMHandle]

        # int rkllm_run(LLMHandle handle, RKLLMInput* rkllm_input, RKLLMInferParam* rkllm_infer_params, void* userdata);
        self.lib.rkllm_run.restype = ctypes.c_int
        self.lib.rkllm_run.argtypes = [
            LLMHandle,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p # userdata
        ]

        # int rkllm_run_async(LLMHandle handle, RKLLMInput* rkllm_input, RKLLMInferParam* rkllm_infer_params, void* userdata);
        # Assuming async also takes userdata for the callback context
        self.lib.rkllm_run_async.restype = ctypes.c_int
        self.lib.rkllm_run_async.argtypes = [
            LLMHandle,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p # userdata
        ]

        # int rkllm_abort(LLMHandle handle);
        self.lib.rkllm_abort.restype = ctypes.c_int
        self.lib.rkllm_abort.argtypes = [LLMHandle]

        # int rkllm_is_running(LLMHandle handle);
        self.lib.rkllm_is_running.restype = ctypes.c_int # 0 if running, non-zero otherwise
        self.lib.rkllm_is_running.argtypes = [LLMHandle]

        # int rkllm_clear_kv_cache(LLMHandle handle, int keep_system_prompt, int* start_pos, int* end_pos);
        self.lib.rkllm_clear_kv_cache.restype = ctypes.c_int
        self.lib.rkllm_clear_kv_cache.argtypes = [
            LLMHandle, 
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),  # start_pos
            ctypes.POINTER(ctypes.c_int)   # end_pos
        ]

        # int rkllm_get_kv_cache_size(LLMHandle handle, int* cache_sizes);
        self.lib.rkllm_get_kv_cache_size.restype = ctypes.c_int
        self.lib.rkllm_get_kv_cache_size.argtypes = [LLMHandle, ctypes.POINTER(ctypes.c_int)]

        # int rkllm_set_chat_template(LLMHandle handle, const char* system_prompt, const char* prompt_prefix, const char* prompt_postfix);
        self.lib.rkllm_set_chat_template.restype = ctypes.c_int
        self.lib.rkllm_set_chat_template.argtypes = [
            LLMHandle,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p
        ]

        # int rkllm_set_function_tools(LLMHandle handle, const char* system_prompt, const char* tools, const char* tool_response_str);
        self.lib.rkllm_set_function_tools.restype = ctypes.c_int
        self.lib.rkllm_set_function_tools.argtypes = [
            LLMHandle,
            ctypes.c_char_p,  # system_prompt
            ctypes.c_char_p,  # tools
            ctypes.c_char_p   # tool_response_str
        ]

        # int rkllm_set_cross_attn_params(LLMHandle handle, RKLLMCrossAttnParam* cross_attn_params);
        self.lib.rkllm_set_cross_attn_params.restype = ctypes.c_int
        self.lib.rkllm_set_cross_attn_params.argtypes = [LLMHandle, ctypes.POINTER(RKLLMCrossAttnParam)]

    def create_default_param(self) -> RKLLMParam:
        """Creates a default RKLLMParam structure."""
        return self.lib.rkllm_createDefaultParam()

    def init(self, param: RKLLMParam, callback_func) -> int:
        """
        Initializes the LLM.
        :param param: RKLLMParam structure.
        :param callback_func: A Python function that matches the signature:
                              def my_callback(result_ptr, userdata_ptr, state_enum):
                                  result = result_ptr.contents # RKLLMResult
                                  # Process result
                                  # userdata can be retrieved if passed during run, or ignored
                                  # state = LLMCallState(state_enum)
        :return: 0 for success, non-zero for failure.
        """
        if not callable(callback_func):
            raise ValueError("callback_func must be a callable Python function.")

        # Keep a reference to the ctypes callback object to prevent it from being garbage collected
        self._c_callback = LLMResultCallback(callback_func)
        
        ret = self.lib.rkllm_init(ctypes.byref(self.llm_handle), ctypes.byref(param), self._c_callback)
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed with error code {ret}")
        return ret

    def load_lora(self, lora_adapter: RKLLMLoraAdapter) -> int:
        """Loads a Lora adapter."""
        ret = self.lib.rkllm_load_lora(self.llm_handle, ctypes.byref(lora_adapter))
        if ret != 0:
            raise RuntimeError(f"rkllm_load_lora failed with error code {ret}")
        return ret

    def load_prompt_cache(self, prompt_cache_path: str) -> int:
        """Loads a prompt cache from a file."""
        c_path = prompt_cache_path.encode('utf-8')
        ret = self.lib.rkllm_load_prompt_cache(self.llm_handle, c_path)
        if ret != 0:
            raise RuntimeError(f"rkllm_load_prompt_cache failed for {prompt_cache_path} with error code {ret}")
        return ret

    def release_prompt_cache(self) -> int:
        """Releases the prompt cache from memory."""
        ret = self.lib.rkllm_release_prompt_cache(self.llm_handle)
        if ret != 0:
            raise RuntimeError(f"rkllm_release_prompt_cache failed with error code {ret}")
        return ret

    def destroy(self) -> int:
        """Destroys the LLM instance and releases resources."""
        if self.llm_handle and self.llm_handle.value: # Check if handle is not NULL
            ret = self.lib.rkllm_destroy(self.llm_handle)
            self.llm_handle = LLMHandle() # Reset handle
            if ret != 0:
                # Don't raise here as it might be called in __del__
                print(f"Warning: rkllm_destroy failed with error code {ret}") 
            return ret
        return 0 # Already destroyed or not initialized

    def run(self, rkllm_input: RKLLMInput, rkllm_infer_params: RKLLMInferParam, userdata=None) -> int:
        """Runs an LLM inference task synchronously."""
        # userdata can be a ctypes.py_object if you want to pass Python objects,
        # then cast to c_void_p. Or simply None.
        if userdata is not None:
            # Store the userdata object to keep it alive during the call
            self._userdata_ref = userdata
            c_userdata = ctypes.cast(ctypes.pointer(ctypes.py_object(userdata)), ctypes.c_void_p)
        else:
            c_userdata = None
        ret = self.lib.rkllm_run(self.llm_handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), c_userdata)
        if ret != 0:
            raise RuntimeError(f"rkllm_run failed with error code {ret}")
        return ret

    def run_async(self, rkllm_input: RKLLMInput, rkllm_infer_params: RKLLMInferParam, userdata=None) -> int:
        """Runs an LLM inference task asynchronously."""
        if userdata is not None:
            # Store the userdata object to keep it alive during the call
            self._userdata_ref = userdata
            c_userdata = ctypes.cast(ctypes.pointer(ctypes.py_object(userdata)), ctypes.c_void_p)
        else:
            c_userdata = None
        ret = self.lib.rkllm_run_async(self.llm_handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), c_userdata)
        if ret != 0:
            raise RuntimeError(f"rkllm_run_async failed with error code {ret}")
        return ret

    def abort(self) -> int:
        """Aborts an ongoing LLM task."""
        ret = self.lib.rkllm_abort(self.llm_handle)
        if ret != 0:
            raise RuntimeError(f"rkllm_abort failed with error code {ret}")
        return ret

    def is_running(self) -> bool:
        """Checks if an LLM task is currently running. Returns True if running."""
        # The C API returns 0 if running, non-zero otherwise.
        # This is a bit counter-intuitive for a boolean "is_running".
        return self.lib.rkllm_is_running(self.llm_handle) == 0

    def clear_kv_cache(self, keep_system_prompt: bool, start_pos: list = None, end_pos: list = None) -> int:
        """
        清除键值缓存
        
        此函数用于清除部分或全部KV缓存。
        
        参数：
        - keep_system_prompt: 是否在缓存中保留系统提示（True保留，False清除）
                              如果提供了特定范围[start_pos, end_pos)，此标志将被忽略
        - start_pos: 要清除的KV缓存范围的起始位置数组（包含），每个批次一个
        - end_pos: 要清除的KV缓存范围的结束位置数组（不包含），每个批次一个
                   如果start_pos和end_pos都设置为None，将清除整个缓存，keep_system_prompt将生效
                   如果start_pos[i] < end_pos[i]，只有指定的范围会被清除，keep_system_prompt将被忽略
        
        注意：start_pos或end_pos只有在keep_history == 0且生成已通过在回调中返回1暂停时才有效
        
        返回：0表示缓存清除成功，非零表示失败
        """
        # 准备C数组参数
        c_start_pos = None
        c_end_pos = None
        
        if start_pos is not None and end_pos is not None:
            if len(start_pos) != len(end_pos):
                raise ValueError("start_pos和end_pos数组长度必须相同")
            
            # 创建C数组
            c_start_pos = (ctypes.c_int * len(start_pos))(*start_pos)
            c_end_pos = (ctypes.c_int * len(end_pos))(*end_pos)
        
        ret = self.lib.rkllm_clear_kv_cache(
            self.llm_handle, 
            ctypes.c_int(1 if keep_system_prompt else 0),
            c_start_pos,
            c_end_pos
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_clear_kv_cache失败，错误代码：{ret}")
        return ret

    def set_chat_template(self, system_prompt: str, prompt_prefix: str, prompt_postfix: str) -> int:
        """Sets the chat template for the LLM."""
        c_system = system_prompt.encode('utf-8') if system_prompt else b""
        c_prefix = prompt_prefix.encode('utf-8') if prompt_prefix else b""
        c_postfix = prompt_postfix.encode('utf-8') if prompt_postfix else b""
        
        ret = self.lib.rkllm_set_chat_template(self.llm_handle, c_system, c_prefix, c_postfix)
        if ret != 0:
            raise RuntimeError(f"rkllm_set_chat_template failed with error code {ret}")
        return ret

    def get_kv_cache_size(self, n_batch: int) -> list:
        """
        获取给定LLM句柄的键值缓存当前大小
        
        此函数返回当前存储在模型KV缓存中的位置总数。
        
        参数：
        - n_batch: 批次数量，用于确定返回数组的大小
        
        返回：
        - list: 每个批次的缓存大小列表
        """
        # 预分配数组以存储每个批次的缓存大小
        cache_sizes = (ctypes.c_int * n_batch)()
        
        ret = self.lib.rkllm_get_kv_cache_size(self.llm_handle, cache_sizes)
        if ret != 0:
            raise RuntimeError(f"rkllm_get_kv_cache_size失败，错误代码：{ret}")
        
        # 转换为Python列表
        return [cache_sizes[i] for i in range(n_batch)]

    def set_function_tools(self, system_prompt: str, tools: str, tool_response_str: str) -> int:
        """
        为LLM设置函数调用配置，包括系统提示、工具定义和工具响应token
        
        参数：
        - system_prompt: 定义语言模型上下文或行为的系统提示
        - tools: JSON格式的字符串，定义可用的函数，包括它们的名称、描述和参数
        - tool_response_str: 用于识别对话中函数调用结果的唯一标签。它作为标记标签，
                            允许分词器将工具输出与正常对话轮次分开识别
        
        返回：0表示配置设置成功，非零表示错误
        """
        c_system = system_prompt.encode('utf-8') if system_prompt else b""
        c_tools = tools.encode('utf-8') if tools else b""
        c_tool_response = tool_response_str.encode('utf-8') if tool_response_str else b""
        
        ret = self.lib.rkllm_set_function_tools(self.llm_handle, c_system, c_tools, c_tool_response)
        if ret != 0:
            raise RuntimeError(f"rkllm_set_function_tools失败，错误代码：{ret}")
        return ret

    def set_cross_attn_params(self, cross_attn_params: RKLLMCrossAttnParam) -> int:
        """
        为LLM解码器设置交叉注意力参数
        
        参数：
        - cross_attn_params: 包含用于交叉注意力的编码器相关输入数据的结构体
                            （详见RKLLMCrossAttnParam说明）
        
        返回：0表示参数设置成功，非零表示错误
        """
        ret = self.lib.rkllm_set_cross_attn_params(self.llm_handle, ctypes.byref(cross_attn_params))
        if ret != 0:
            raise RuntimeError(f"rkllm_set_cross_attn_params失败，错误代码：{ret}")
        return ret

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    def __del__(self):
        self.destroy() # Ensure resources are freed if object is garbage collected

# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    # This is a placeholder for how you might use it.
    # You'll need a valid .rkllm model and librkllmrt.so in your path.

    # Global list to store results from callback for demonstration
    results_buffer = []

    def my_python_callback(result_ptr, userdata_ptr, state_enum):
        """
        回调函数，由C库调用来处理LLM结果
        
        参数：
        - result_ptr: 指向LLM结果的指针
        - userdata_ptr: 用户数据指针
        - state_enum: LLM调用状态枚举值
        
        返回：
        - 0: 继续推理
        - 1: 暂停推理
        """
        global results_buffer
        state = LLMCallState(state_enum)
        result = result_ptr.contents

        current_text = ""
        if result.text: # 检查char_p是否不为NULL
            current_text = result.text.decode('utf-8', errors='ignore')
        
        print(f"回调: State={state.name}, TokenID={result.token_id}, Text='{current_text}'")
        
        # 显示性能统计信息
        if result.perf.prefill_tokens > 0 or result.perf.generate_tokens > 0:
            print(f"  性能统计: 预填充={result.perf.prefill_tokens}tokens/{result.perf.prefill_time_ms:.1f}ms, "
                  f"生成={result.perf.generate_tokens}tokens/{result.perf.generate_time_ms:.1f}ms, "
                  f"内存={result.perf.memory_usage_mb:.1f}MB")
        
        results_buffer.append(current_text)

        if state == LLMCallState.RKLLM_RUN_FINISH:
            print("推理完成。")
        elif state == LLMCallState.RKLLM_RUN_ERROR:
            print("推理错误。")
        
        # 返回0继续推理，返回1暂停推理
        return 0

    # --- Attempt to use the wrapper ---
    try:
        print("Initializing RKLLMRuntime...")
        # Adjust library_path if librkllmrt.so is not in default search paths
        # e.g., library_path="./path/to/librkllmrt.so"
        rk_llm = RKLLMRuntime() 

        print("Creating default parameters...")
        params = rk_llm.create_default_param()

        # --- Configure parameters ---
        # THIS IS CRITICAL: model_path must point to an actual .rkllm file
        # For this example to run, you need a model file.
        # Let's assume a dummy path for now, this will fail at init if not valid.
        model_file = "dummy_model.rkllm" 
        if not os.path.exists(model_file):
            print(f"Warning: Model file '{model_file}' does not exist. Init will likely fail.")
            # Create a dummy file for the example to proceed further, though init will still fail
            # with a real library unless it's a valid model.
            with open(model_file, "w") as f:
                f.write("dummy content")

        params.model_path = model_file.encode('utf-8')
        params.max_context_len = 512
        params.max_new_tokens = 128
        params.top_k = 1 # Greedy
        params.temperature = 0.7
        params.repeat_penalty = 1.1
        # ... set other params as needed

        print(f"Initializing LLM with model: {params.model_path.decode()}...")
        # This will likely fail if dummy_model.rkllm is not a valid model recognized by the library
        try:
            rk_llm.init(params, my_python_callback)
            print("LLM Initialized.")
        except RuntimeError as e:
            print(f"Error during LLM initialization: {e}")
            print("This is expected if 'dummy_model.rkllm' is not a valid model.")
            print("Replace 'dummy_model.rkllm' with a real model path to test further.")
            exit()


        # --- Prepare input ---
        print("准备输入...")
        rk_input = RKLLMInput()
        rk_input.role = b"user"  # 设置角色为用户输入
        rk_input.enable_thinking = False  # 禁用思考模式（适用于Qwen3模型）
        rk_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        
        prompt_text = "将以下英文文本翻译成中文：'Hello, world!'"
        c_prompt = prompt_text.encode('utf-8')
        rk_input._union_data.prompt_input = c_prompt # 直接访问联合体成员

        # --- Prepare inference parameters ---
        print("Preparing inference parameters...")
        infer_params = RKLLMInferParam()
        infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        infer_params.keep_history = 1 # True
        # infer_params.lora_params = None # or set up RKLLMLoraParam if using LoRA
        # infer_params.prompt_cache_params = None # or set up RKLLMPromptCacheParam

        # --- Run inference ---
        print(f"Running inference with prompt: '{prompt_text}'")
        results_buffer.clear()
        try:
            rk_llm.run(rk_input, infer_params) # Userdata is None by default
            print("\n--- Full Response ---")
            print("".join(results_buffer))
            print("---------------------\n")
        except RuntimeError as e:
            print(f"Error during LLM run: {e}")


        # --- Example: Set chat template (if model supports it) ---
        # print("Setting chat template...")
        # try:
        #     rk_llm.set_chat_template("You are a helpful assistant.", "<user>: ", "<assistant>: ")
        #     print("Chat template set.")
        # except RuntimeError as e:
        #     print(f"Error setting chat template: {e}")

        # --- Example: Clear KV Cache ---
        # print("Clearing KV cache (keeping system prompt if any)...")
        # try:
        #     rk_llm.clear_kv_cache(keep_system_prompt=True)
        #     print("KV cache cleared.")
        # except RuntimeError as e:
        #     print(f"Error clearing KV cache: {e}")

        # --- 示例：获取KV缓存大小 ---
        # print("获取KV缓存大小...")
        # try:
        #     cache_sizes = rk_llm.get_kv_cache_size(n_batch=1)  # 假设批次大小为1
        #     print(f"当前KV缓存大小: {cache_sizes}")
        # except RuntimeError as e:
        #     print(f"获取KV缓存大小错误: {e}")

        # --- 示例：设置函数工具 ---
        # print("设置函数调用工具...")
        # try:
        #     system_prompt = "你是一个有用的助手，可以调用提供的函数来帮助用户。"
        #     tools = '''[{
        #         "name": "get_weather",
        #         "description": "获取指定城市的天气信息",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "city": {"type": "string", "description": "城市名称"}
        #             },
        #             "required": ["city"]
        #         }
        #     }]'''
        #     tool_response_str = "<tool_response>"
        #     rk_llm.set_function_tools(system_prompt, tools, tool_response_str)
        #     print("函数工具设置成功。")
        # except RuntimeError as e:
        #     print(f"设置函数工具错误: {e}")

        # --- 示例：清除KV缓存（带范围参数） ---
        # print("使用范围参数清除KV缓存...")
        # try:
        #     # 清除位置10到20的缓存
        #     start_positions = [10]  # 批次0的起始位置
        #     end_positions = [20]    # 批次0的结束位置
        #     rk_llm.clear_kv_cache(keep_system_prompt=True, start_pos=start_positions, end_pos=end_positions)
        #     print("范围KV缓存清除完成。")
        # except RuntimeError as e:
        #     print(f"清除范围KV缓存错误: {e}")
        
    except OSError as e:
        print(f"OSError: {e}. Could not load the RKLLM library.")
        print("Please ensure 'librkllmrt.so' is in your LD_LIBRARY_PATH or provide the full path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'rk_llm' in locals() and rk_llm.llm_handle and rk_llm.llm_handle.value:
            print("Destroying LLM instance...")
            rk_llm.destroy()
            print("LLM instance destroyed.")
        if os.path.exists(model_file) and model_file == "dummy_model.rkllm":
             os.remove(model_file) # Clean up dummy file

    print("Example finished.")