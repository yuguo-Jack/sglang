"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, Optional, Tuple
import os
import re
import torch
from torch import nn
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import pad_weight, gemm_bank_conf
from vllm import _custom_ops as ops
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

Qwen2Config = None


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )
        self.quant_method = None
        if quant_config is not None:
            self.quant_method=quant_config.get_name()

        self.use_fa_pad = os.environ.get('FA_PAD') == '1'

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.use_fa_pad and self.quant_method is None:
            qkv = qkv[...,:-32]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, i, quant_config=quant_config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2Model(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        self.quant_method = None
        if quant_config is not None:
            self.quant_method=quant_config.get_name()

        self.use_gemm_nn = True
        self.use_gemm_pad = os.environ.get('GEMM_PAD') == '1'
        self.use_fa_pad = os.environ.get('FA_PAD') == '1'

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head.weight, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                if (
                    self.config.tie_word_embeddings
                    and name == "model.embed_tokens.weight"
                ):
                    weight_loader(params_dict["lm_head.weight"], loaded_weight)
        if self.use_gemm_nn and self.quant_method is None:
            lay_key_words = [
                "self_attn.qkv_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_up_proj.weight",
                "mlp.down_proj.weight",
            ]
            combined_words = "|".join(lay_key_words)
            
            lay_qkv_words = ["self_attn.qkv_proj.weight"]   
            qkv_words = "|".join(lay_qkv_words)  
            
            lay_qkv_bias_words = ["self_attn.qkv_proj.bias"]   
            qkv_bias_words = "|".join(lay_qkv_bias_words) 
            
            for layername, weight in params_dict.items():
                if self.use_fa_pad and (re.findall(qkv_bias_words, layername)):
                    weight.data = pad_weight(weight.data, 32)
                    
                matches = re.findall(combined_words, layername)
                if matches:    
                    if self.use_gemm_pad and gemm_bank_conf(weight.data.shape[0]):
                        weight.data = pad_weight(weight.data, 32)  
                           
                    if self.use_fa_pad and (re.findall(qkv_words, layername)):
                        if not gemm_bank_conf(weight.data.shape[0]):
                            weight.data = pad_weight(weight.data, 32)
                        
                    _weight = torch.zeros_like(weight.data)
                    ori_shape =_weight.shape
                    
                    ops.trans_w16_gemm(_weight, weight.data, _weight.shape[0], _weight.shape[1])
                    weight.data.copy_(_weight)
                    
                    weight.data=weight.data.reshape(ori_shape[1],-1)
                    
        if self.quant_method == "awq":
            lay_key_words = [
                "self_attn.qkv_proj.qweight",
                "self_attn.o_proj.qweight",
                "mlp.gate_up_proj.qweight",
                "mlp.down_proj.qweight"
            ]
            combined_words = "|".join(lay_key_words)
            
            for layername, weight in params_dict.items():
                
                matches = re.findall(combined_words, layername)
                if matches:
                    qweight =params_dict[layername]
                    qzeros=params_dict[layername.replace("qweight", "qzeros")]
                    scales=params_dict[layername.replace("qweight", "scales")]
                    zeros_and_scalse =params_dict[layername.replace("qweight", "zeros_and_scales")]
                    
                    group_size= self.quant_config.group_size 
                   
                    dim_n = scales.data.shape[1]
                    dim_k = qweight.data.shape[0]
                    pad_group=2              
                    
                    _qw, _sz=ops.convert_s4(qweight,qzeros,scales,int(group_size)) 
                    
                    sz = ops.sz_permute(_sz).reshape(-1,dim_n)       
                    
                    zeros_and_scalse.data.copy_(sz)
                    qweight.data.copy_(_qw)
                    
                    #reshape
                    zeros_and_scalse.data=zeros_and_scalse.reshape(dim_n,-1)    #[k/greop_size,n]------>[n,k/group_size]
                    qweight.data=qweight.data.reshape(dim_n,-1)                      #[k,n/8]---->[n,k/8]  
                
                    if dim_k % 4096==0:
                        zeros_and_scalse_pad= torch.zeros(dim_n,pad_group,dtype=torch.int32).cuda()
                        zeros_and_scalse.data=torch.cat((zeros_and_scalse.data,zeros_and_scalse_pad),dim=1).contiguous()
                        qweight_pad= torch.zeros(dim_n,int(group_size//4),dtype=torch.int32).cuda()
                        qweight.data=torch.cat((qweight.data,qweight_pad),dim=1).contiguous()    


EntryClass = Qwen2ForCausalLM
