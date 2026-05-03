# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, Optional, Union

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention as HFGptOssAttention,
    GptOssForCausalLM as HFGptOssForCausalLM,
    GptOssModel as HFGptOssModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
    load_balancing_loss_func,
)


def _resolve_past_seen_tokens(
    past_key_values: Optional[Union[list, object]],
) -> int:
    if past_key_values is None:
        return 0

    if isinstance(past_key_values, list):
        try:
            return int(past_key_values[0][0].current_length.item())
        except Exception as exc:  # noqa: B902
            raise ValueError("Unsupported custom KV layout for GPT-OSS") from exc

    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())

    raise ValueError(f"Unsupported past_key_values type: {type(past_key_values)}")


def _build_causal_mask(
    batch_size: int,
    q_len: int,
    kv_len: int,
    past_seen_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    min_dtype = torch.finfo(dtype).min
    key_positions = torch.arange(kv_len, device=device).view(1, kv_len)
    query_positions = torch.arange(q_len, device=device).view(q_len, 1) + past_seen_tokens

    blocked = key_positions > query_positions
    if sliding_window is not None and int(sliding_window) > 0:
        low = query_positions - int(sliding_window) + 1
        blocked = blocked | (key_positions < low)

    mask_2d = torch.zeros((q_len, kv_len), dtype=dtype, device=device)
    mask_2d.masked_fill_(blocked, min_dtype)

    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, kv_len).clone()

    if attention_mask is not None and attention_mask.dim() == 2:
        attn = attention_mask.to(device=device)
        if attn.shape[1] < kv_len:
            pad = torch.ones(
                attn.shape[0],
                kv_len - attn.shape[1],
                device=device,
                dtype=attn.dtype,
            )
            attn = torch.cat([pad, attn], dim=1)
        elif attn.shape[1] > kv_len:
            attn = attn[:, -kv_len:]
        pad_mask = (1.0 - attn.to(dtype)).unsqueeze(1).unsqueeze(1) * min_dtype
        mask_4d = mask_4d + pad_mask

    return mask_4d


def _apply_tree_mask_if_any(
    mask: torch.Tensor,
    tree_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if tree_mask is None:
        return mask

    _, _, tree_q, tree_k = tree_mask.shape
    min_dtype = torch.finfo(mask.dtype).min
    mask = mask.clone()
    mask[:, :, -tree_q:, -tree_k:][tree_mask == 0] = min_dtype
    return mask


class GptOssAttention(HFGptOssAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            if isinstance(past_key_value, list):
                past_key, past_value = past_key_value[self.layer_idx]
                key_states = past_key.cat(key_states)
                value_states = past_value.cat(value_states)
            else:
                cache_kwargs = {"cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GptOssModel(HFGptOssModel):
    def __init__(self, config):
        super().__init__(config)
        self.eagle_aux_hidden_state_layer_ids = None

        for i, layer in enumerate(self.layers):
            layer.self_attn = GptOssAttention(config=config, layer_idx=i)

    def set_eagle_aux_hidden_state_layer_ids(self, layer_ids: Optional[List[int]]) -> None:
        self.eagle_aux_hidden_state_layer_ids = layer_ids

    def _resolve_aux_layer_ids(self) -> List[int]:
        if self.eagle_aux_hidden_state_layer_ids is not None:
            return [int(x) for x in self.eagle_aux_hidden_state_layer_ids]

        num_layers = len(self.layers)
        # Keep parity with training-time default (layer ids without embedding
        # offset): [1, mid-1, last-4]. In inference we map with +1.
        return [1, max(0, num_layers // 2 - 1), max(0, num_layers - 4)]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list, object]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = _resolve_past_seen_tokens(past_key_values)
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        batch_size, q_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
        kv_len = past_seen_tokens + q_len
        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            full_mask = _build_causal_mask(
                batch_size=batch_size,
                q_len=q_len,
                kv_len=kv_len,
                past_seen_tokens=past_seen_tokens,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                attention_mask=attention_mask,
                sliding_window=None,
            )
            sliding_mask = _build_causal_mask(
                batch_size=batch_size,
                q_len=q_len,
                kv_len=kv_len,
                past_seen_tokens=past_seen_tokens,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                attention_mask=attention_mask,
                sliding_window=self.config.sliding_window,
            )

            tree_mask = getattr(self, "tree_mask", None)
            causal_mask_mapping = {
                "full_attention": _apply_tree_mask_if_any(full_mask, tree_mask),
                "sliding_attention": _apply_tree_mask_if_any(sliding_mask, tree_mask),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        aux_ids = self._resolve_aux_layer_ids()
        aux_decoder_ids = [layer_id + 1 for layer_id in aux_ids]
        aux_decoder_ids = [
            layer_id for layer_id in aux_decoder_ids if layer_id >= 0 and layer_id < len(self.layers)
        ]
        aux_id_set = set(aux_decoder_ids)

        all_hidden_states = ()
        for layer_idx, decoder_layer in enumerate(self.layers):
            if layer_idx in aux_id_set:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class GptOssForCausalLM(HFGptOssForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = GptOssModel(config)

    def set_eagle_aux_hidden_state_layer_ids(self, layer_ids: Optional[List[int]]) -> None:
        self.model.set_eagle_aux_hidden_state_layer_ids(layer_ids)

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list, object]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None and aux_loss is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
