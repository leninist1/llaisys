from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType
from ..libllaisys import llaisysDeviceType_t
from ..libllaisys.models import LlaisysQwen2Meta

from pathlib import Path
import safetensors
import json
import ctypes


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self._device = device
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        # read model config from config.json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # parse dim
        hs = int(config["hidden_size"])                
        nlayer = int(config["num_hidden_layers"])      
        nh = int(config["num_attention_heads"])        
        nkvh = int(config.get("num_key_value_heads", nh))  
        di = int(config["intermediate_size"])          
        dh = int(hs // nh)                             
        
        # parse key params
        maxseq = int(config["max_position_embeddings"])
        voc = int(config["vocab_size"])                
        epsilon = float(config["rms_norm_eps"])  
        theta = float(config["rope_theta"])   
        end_token = int(config["eos_token_id"])
        

        # construct C struct LlaisysQwen2Meta
        meta = LlaisysQwen2Meta(
            dtype=DataType.BF16,
            nlayer=nlayer,
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=dh,
            di=di,
            maxseq=maxseq,
            voc=voc,
            epsilon=epsilon,
            theta=theta,
            end_token=end_token,
        )

        # only use cpu
        device_ids = (ctypes.c_int * 1)(0)
        # create model instance
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), llaisysDeviceType_t(device), device_ids, 1
        )

        # get model weights
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        self._end_token = end_token

        # traverse all safetensors files, in fact only one file in qwen2
        for file in sorted(model_path.glob("*.safetensors")):
            # load on cpu, I use pt framework to load bfloat16 weights here
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                weight = self._match_weight(name_)
                if weight is None:
                    continue
                # load weight to c side
                arr = data_.get_tensor(name_).contiguous() #c-contiguous
                
                LIB_LLAISYS.tensorLoad(weight, ctypes.c_void_p(arr.data_ptr()))

    
    def _match_weight(self, name: str):
        # match weight name to c struct field

        w = self._weights
        # input embedding
        if name == "model.embed_tokens.weight":
            return w.in_embed
        # output embedding
        if name in ("lm_head.weight", "model.lm_head.weight"):
            return w.out_embed
        # final LayerNorm
        if name == "model.norm.weight":
            return w.out_norm_w
        # only processtransformer layer weights
        if not name.startswith("model.layers."):
            return None
        parts = name.split(".")
        if len(parts) < 5:
            return None
        layer = int(parts[2])      # 提取层索引
        tail = ".".join(parts[3:])  # 剩余后缀
        # Attention Layer
        if tail == "input_layernorm.weight":
            return w.attn_norm_w[layer]
        if tail == "self_attn.q_proj.weight":
            return w.attn_q_w[layer]
        if tail == "self_attn.q_proj.bias":
            return w.attn_q_b[layer]
        if tail == "self_attn.k_proj.weight":
            return w.attn_k_w[layer]
        if tail == "self_attn.k_proj.bias":
            return w.attn_k_b[layer]
        if tail == "self_attn.v_proj.weight":
            return w.attn_v_w[layer]
        if tail == "self_attn.v_proj.bias":
            return w.attn_v_b[layer]
        if tail == "self_attn.o_proj.weight":
            return w.attn_o_w[layer]
        # FFN layer
        if tail == "post_attention_layernorm.weight":
            return w.mlp_norm_w[layer]
        if tail == "mlp.gate_proj.weight":
            return w.mlp_gate_w[layer]
        if tail == "mlp.up_proj.weight":
            return w.mlp_up_w[layer]
        if tail == "mlp.down_proj.weight":
            return w.mlp_down_w[layer]
        return None

    def _infer(self, tokens: Sequence[int]) -> int:
        # step forward infer

        if len(tokens) == 0:
            return self._end_token
        # convert python list to c int64 array
        arr = (ctypes.c_int64 * len(tokens))(*tokens)
        return int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, arr, ctypes.c_size_t(len(tokens))
            )
        )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # max new tokens default value:32
        if max_new_tokens is None:
            max_new_tokens = 32 
        tokens = list(inputs)
        if max_new_tokens == 0:
            return tokens
        # prefill
        next_token = self._infer(tokens)
        tokens.append(next_token)
        # decode
        for _ in range(max_new_tokens - 1):
            if tokens[-1] == self._end_token:
                break
            next_token = self._infer([tokens[-1]])
            tokens.append(next_token)
        return tokens
