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

"""Memory pool."""

import logging
from typing import List, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(self, size: int, max_context_len: int, device: str):
        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token = torch.empty(
            (size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


class BaseTokenToKVPool:
    """A memory pool that maps a token to its kv cache locations"""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        if dtype == torch.float8_e5m2:
            # NOTE: Store as torch.uint8 because Tensor index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype

        self.free_slots = None
        self.clear()

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return torch.tensor(select_index, dtype=torch.int32, device=self.device)

    def free(self, free_index: torch.Tensor):
        self.free_slots = np.concatenate((self.free_slots, free_index.cpu().numpy()))

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots = np.arange(1, self.size + 1)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def get_key_scales_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_value_scales_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_kv_scales_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class MHATokenToKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        kv_cache_dtype_str: str,
        kvint4_groupsize: int,
    ):
        super().__init__(size, dtype, device)

        self.kv_cache_dtype_str = kv_cache_dtype_str
        if kv_cache_dtype_str == "int4":
            # [size, head_num, head_dim] for each layer
            self.k_buffer = [
                torch.empty(
                    (size + 1, head_num, head_dim // 2), dtype=torch.int8, device="cuda"
                )
                for _ in range(layer_num)
            ]
            self.v_buffer = [
                torch.empty(
                    (size + 1, head_num, head_dim // 2), dtype=torch.int8, device="cuda"
                )
                for _ in range(layer_num)
            ]
        else:
            # [size, head_num, head_dim] for each layer
            self.k_buffer = [
                torch.empty(
                    (size + 1, head_num, head_dim), dtype=self.store_dtype, device="cuda"
                )
                for _ in range(layer_num)
            ]
            self.v_buffer = [
                torch.empty(
                    (size + 1, head_num, head_dim), dtype=self.store_dtype, device="cuda"
                )
                for _ in range(layer_num)
            ]
        if dtype == torch.int8:
            if kv_cache_dtype_str == "int4":
                self.quant_group_size = kvint4_groupsize
                self.k_scales_buffer = [
                    torch.empty(
                        (size + 1, head_num, 128 // self.quant_group_size), dtype=torch.float16, device="cuda"
                    )
                    for _ in range(layer_num)
                ]
                self.v_scales_buffer = [
                    torch.empty(
                        (size + 1, head_num, 128 // self.quant_group_size), dtype=torch.float16, device="cuda"
                    )
                    for _ in range(layer_num)
                ]
            else:
                self.k_scales_buffer = [
                    torch.empty(
                        (size + 1, head_num, 1), dtype=torch.float16, device="cuda"
                    )
                    for _ in range(layer_num)
                ]
                self.v_scales_buffer = [
                    torch.empty(
                        (size + 1, head_num, 1), dtype=torch.float16, device="cuda"
                    )
                    for _ in range(layer_num)
                ]

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)
    
    def get_key_scales_buffer(self, layer_id: int):
        if self.dtype == torch.int8:
            return self.k_scales_buffer[layer_id]
        else:
            return None

    def get_value_scales_buffer(self, layer_id: int):
        if self.dtype == torch.int8:
            return self.v_scales_buffer[layer_id]
        else:
            return None

    def get_kv_scales_buffer(self, layer_id: int):
        if self.dtype == torch.int8:
            return self.get_key_scales_buffer(layer_id), self.get_value_scales_buffer(layer_id)
        else:
            return None

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        if self.dtype == torch.int8:
            if self.kv_cache_dtype_str == "int4":
                from sglang.srt.layers.attention.triton_ops.decode_attention_int4kv import (
                    destindex_copy_quantize_int4kv,
                )
                destindex_copy_quantize_int4kv(cache_k, loc, self.k_buffer[layer_id], self.k_scales_buffer[layer_id], self.quant_group_size)
                destindex_copy_quantize_int4kv(cache_v, loc, self.v_buffer[layer_id], self.v_scales_buffer[layer_id], self.quant_group_size)
            else:
                from sglang.srt.layers.attention.triton_ops.decode_attention_int8kv import (
                    destindex_copy_quantize_kv,
                )
                destindex_copy_quantize_kv(cache_k, loc, self.k_buffer[layer_id], self.k_scales_buffer[layer_id])
                destindex_copy_quantize_kv(cache_v, loc, self.v_buffer[layer_id], self.v_scales_buffer[layer_id])
        else:
            if cache_k.dtype != self.dtype:
                cache_k = cache_k.to(self.dtype)
            if cache_v.dtype != self.dtype:
                cache_v = cache_v.to(self.dtype)
            if self.store_dtype != self.dtype:
                self.k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
                self.v_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
            else:
                self.k_buffer[layer_id][loc] = cache_k
                self.v_buffer[layer_id][loc] = cache_v


class MLATokenToKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
    ):
        super().__init__(size, dtype, device)

        self.kv_lora_rank = kv_lora_rank
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.kv_buffer = [
            torch.empty(
                (size + 1, 1, kv_lora_rank + qk_rope_head_dim),
                dtype=self.store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id].view(self.dtype)
        return self.kv_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id][..., : self.kv_lora_rank].view(self.dtype)
        return self.kv_buffer[layer_id][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
        else:
            self.kv_buffer[layer_id][loc] = cache_k
