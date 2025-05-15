from functools import wraps
from typing import List, Optional, Tuple, Union
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb
# from transformers.models.llama.modeling_llama import repeat_kv
from dotenv import load_dotenv

from transformers.utils import logging

# config logger
logger = logging.get_logger(__name__)

# env
load_dotenv()
ACCESS_TOKEN=os.getenv("ACCESS_TOKEN")


global g_llama_sdpa_attn_forward_orgn, g_mistral_sdpa_attn_forward_orgn, g_falcon_sdpa_attn_forward_orgn
g_llama_sdpa_attn_forward_orgn = transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
g_mistral_sdpa_attn_forward_orgn = transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward
g_falcon_sdpa_attn_forward_orgn = transformers.models.falcon.modeling_falcon.FalconAttention.forward



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    """
    # batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    shape_ = list(hidden_states.shape)
    shape_n = shape_.copy()
    shape_n.insert(2, n_rep)
    shape_[1] *= n_rep
    hidden_states = hidden_states[:, :, None, ...].expand(*shape_n)
    return hidden_states.reshape(*shape_)


def de_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    shape_ = list(hidden_states.shape)
    shape_n = shape_.copy()
    shape_n.insert(2, n_rep)
    shape_n[1] //= n_rep
    hidden_states = hidden_states.reshape(*shape_n).sum(axis=2)
    return hidden_states


def score_scaled_dot_product_attention(
        query: torch.Tensor ,
        key: torch.Tensor ,
        value: torch.Tensor ,
        attn_mask:  torch.Tensor = None,
        pos_weight:  torch.Tensor = None,
        scale_factor: float = None,
        shrink_factor: float = None,
        init_f: bool = False,
        window_size: int = None,
        window_pool: Optional[str] = None,
        kernel_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def sub_sdpa(
            t_bias: int,
            sub_query: torch.Tensor ,
            key: torch.Tensor ,
            value: torch.Tensor ,
            sub_attn_mask:  torch.Tensor = None,
            pos_weight:  torch.Tensor = None,
            scale_factor: float = None,
            shrink_factor: float = None,
            need_mask=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = sub_query.dtype
        bzs, n_head, t_k, d_k = key.size()
        bzs, n_head, t_q, d_q = sub_query.size()
        scale = 1.0 / torch.sqrt(torch.tensor(d_k, dtype=dtype))
        
        if sub_attn_mask is None and  need_mask:
            mask_dtype = torch.float32
            sub_attn_mask = torch.triu(torch.full((t_q, t_k), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=key.device), diagonal=t_k - t_bias + 1)
            sub_attn_mask = sub_attn_mask[None, None, ...]
        scores = torch.matmul(sub_query, key.transpose(-2, -1))
        scores = scores * scale
        if sub_attn_mask is not None:
            scores = scores + sub_attn_mask

        # softmax
        if pos_weight is not None:
            scores += scale_factor * torch.log(pos_weight)[..., None, :]
        attn_weights = F.softmax(scores, dim=-1, dtype=value.dtype)

        output = torch.matmul(attn_weights, value)
        if shrink_factor < 1 and t_q > 1:
            shrink_f = torch.pow(shrink_factor, torch.arange(t_q, device=attn_weights.device))
            shrink_f_flipped = torch.flip(shrink_f, dims=[0])
            score = (attn_weights * shrink_f_flipped[None, None, :, None]).sum(dim=-2)
        else:
            score = attn_weights.sum(dim=-2)
        return output, score
    
    bzs, n_head, t_k, d_k = key.size()
    bzs, n_head, t_q, d_q = query.size()

    if init_f:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        )
    else:
        attn_output = None
    
    if window_size is not None:
        query = query[:, :, -window_size:, :]
        bzs, n_head, t_q, d_q = query.size()
        attn_mask = attn_mask[:, :, -window_size:, :] if attn_mask is not None else None
    need_mask =  t_q > 1
    sub_len = t_q
    shrink_factor_tensor = torch.tensor(shrink_factor, dtype=key.dtype, device=key.device)
    while 1:
        try:
            score = torch.zeros(bzs, n_head, t_k, dtype=key.dtype, device=key.device)
            output = []
            for start_ind in range(0, t_q, sub_len):
                sub_query = query[:, :, start_ind : start_ind + sub_len]
                sub_attn_mask = attn_mask[:, :, start_ind : start_ind + sub_len] if attn_mask else None
                output_sub, score_ = sub_sdpa(t_q - start_ind, sub_query, key, value, sub_attn_mask, pos_weight, scale_factor, shrink_factor, need_mask=need_mask)
                output.append(output_sub)
                score *= torch.pow(shrink_factor_tensor, sub_query.size(2))
                score += score_
            else:
                output = torch.concat(output, axis=2)
                break
        except torch.OutOfMemoryError as e:
            if sub_len<= 10:
                raise e
            sub_len = (sub_len + 3) // 4
    if init_f:
        assert attn_output is not None
        output = attn_output
        if window_pool == 'avgpool':
            score = F.avg_pool1d(score, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
        elif window_pool == 'maxpool':
            score = F.max_pool1d(score, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    return output, score


def topk_split(E, S, k):
    b, t, d = E.shape
    # Step 1: Get top K indices
    topk_values, topk_indices = torch.topk(S, k, dim=1)

    # Step 2: Gather top K elements for tensor A
    A = torch.gather(E, 1, topk_indices.unsqueeze(-1).expand(-1, -1, d))

    # Step 3: Create a mask for indices not in top K
    all_indices = torch.arange(t, device=S.device).unsqueeze(0).expand(b, -1)
    topk_mask = torch.zeros_like(S, dtype=torch.bool)
    topk_mask.scatter_(1, topk_indices, True)

    # Complement indices for B, preserving order
    complement_indices = all_indices[~topk_mask].view(b, t - k)

    # Step 4: Gather remaining elements for tensor B
    tail_values = torch.gather(S, 1, complement_indices)
    B = torch.gather(E, 1, complement_indices.unsqueeze(-1).expand(-1, -1, d))
    return A, B, topk_values, tail_values
    # A: Tensor of shape [b, k, d]
    # B: Tensor of shape [b, t - k, d]


def cache_init(
        key_states, value_states, **cache_kwargs
    ):
    cache_budget = cache_kwargs["cache_budget"]
    cache_tail = cache_kwargs["cache_tail"]
    cache_dense = cache_kwargs["cache_dense"]
    score_update = cache_kwargs["score_update"]
    # cache_top = 0.5
    scores = cache_kwargs["scores"] # B, H, Tk
    cache_top_size = (cache_budget - cache_tail) - cache_dense

    bsz, n_head, t_k, d_k = key_states.size()
    if t_k <= cache_budget:
        pos_weight = torch.ones_like(scores)
        return key_states, value_states, pos_weight, scores

    key_li, value_li, score_li, weight_li = [], [], [], []
    for key, value, score in zip(key_states, value_states, scores):
        # n_head, t_k, d_k = key.size()
        assert t_k >= cache_budget
        kv_status = torch.cat([key, value], axis=-1)
        kv_status_ = kv_status[..., :t_k-cache_tail, :]
        score_ = score[..., :t_k-cache_tail]
        # torch.Size([32, 0, 256]) torch.Size([32, 0]) 32 13 0
        cache_top, cache_residual, score_top, score_residual = topk_split(
            kv_status_.view(n_head, t_k-cache_tail, -1),
            score_.view(n_head, t_k-cache_tail),
            k=cache_top_size
        )
        merge_size, tail_add = divmod(cache_residual.size(-2), cache_dense)

        # special
        sep_id = tail_add * (merge_size + 1)
        cache_residual_m = torch.cat([
            cache_residual[..., :sep_id, :].view(n_head, tail_add, merge_size + 1, 2 * d_k).sum(axis=-2),
            cache_residual[..., sep_id:, :].view(n_head, cache_dense - tail_add, merge_size, 2 * d_k).sum(axis=-2),
        ], dim=-2)
        if score_update == "max":
            socre_m = torch.cat([
                score_residual[..., :sep_id].view(n_head, tail_add, merge_size + 1).max(axis=-1)[0],
                score_residual[..., sep_id:].view(n_head, cache_dense - tail_add, merge_size).max(axis=-1)[0],
            ], dim=-1)
        elif score_update == "sum":
            socre_m = torch.cat([
                score_residual[..., :sep_id].view(n_head, tail_add, merge_size + 1).sum(axis=-1),
                score_residual[..., sep_id:].view(n_head, cache_dense - tail_add, merge_size).sum(axis=-1),
            ], dim=-1)
        else:
            raise ValueError(f"Invalid args<score_update: {score_update}>")
        weight_m = torch.ones_like(socre_m) * merge_size
        weight_m[..., :tail_add] += 1

        cache_residual_m /= weight_m[..., None]
        # cat
        kv_status_cat = torch.cat([cache_residual_m, cache_top, kv_status[..., t_k-cache_tail:, :]], dim=-2)
        score_cat = torch.cat([socre_m, score_top, score[..., t_k-cache_tail:]], dim=-1)
        weight_cat = torch.cat([weight_m, torch.ones_like(score_top), torch.ones_like(score[..., t_k-cache_tail:])], dim=-1)

        key_li.append(kv_status_cat[..., :d_k])
        value_li.append(kv_status_cat[..., d_k:])
        score_li.append(score_cat)
        weight_li.append(weight_cat)

    key_states = torch.stack(key_li, dim=0)
    value_states = torch.stack(value_li, dim=0)
    scores = torch.stack(score_li, dim=0)
    pos_weight = torch.stack(weight_li, dim=0)
    
    return key_states, value_states, pos_weight, scores


def cache_merge(key_states, value_states, w_n, s_n, key_states_l, value_states_l, pos_weight_l, pos_score_l, **cache_kwargs):
    metric = cache_kwargs["metric"]
    score_update = cache_kwargs["score_update"]
    cache_dense = cache_kwargs["cache_dense"]

    bsz, n_head, cache_t, d_k = key_states_l.size()

    # Find the minimum index for pos_score_l
    _, mrg_i = torch.min(pos_score_l[..., cache_dense:], dim=-1)
    mrg_i += cache_dense

    # Gather the corresponding key_states_l and pos_weight_l
    mrg_i_exp = mrg_i[..., None, None].expand(-1, -1, -1, d_k)
    k_i = torch.gather(key_states_l, -2, mrg_i_exp)

    if metric == "dot_product":
        metric_ij = -torch.einsum("bhkd,bhtd->bht", k_i, key_states_l[..., :cache_dense, :])
    elif metric == "l2":
        metric_ij = torch.square(k_i - key_states_l[..., :cache_dense, :]).sum(axis=-1)
    else:
        raise ValueError(f"Invalid metric: {metric}")
    # metric_ij.scatter_(-1, mrg_i[..., None], torch.finfo(metric_ij.dtype).max)
    _, mrg_j = torch.min(metric_ij, dim=-1)
    # print(mrg_j[])
    mrg_j_exp = mrg_j[..., None, None].expand(-1, -1, -1, d_k)

    # Compute weights and weighted averages for merging
    w_i = torch.gather(pos_weight_l, -1, mrg_i[..., None])
    w_j = torch.gather(pos_weight_l, -1, mrg_j[..., None])
    w_ij = w_i + w_j

    k_j = torch.gather(key_states_l, -2, mrg_j_exp)
    v_i = torch.gather(value_states_l, -2, mrg_i_exp)
    v_j = torch.gather(value_states_l, -2, mrg_j_exp)

    k_ij = k_i.mul_(w_i[..., None]).add_(k_j.mul(w_j[..., None])).div_(w_ij[..., None])
    v_ij = v_i.mul_(w_i[..., None]).add_(v_j.mul(w_j[..., None])).div_(w_ij[..., None])

    # Compute scores
    s_i = torch.gather(pos_score_l, -1, mrg_i[..., None])
    s_j = torch.gather(pos_score_l, -1, mrg_j[..., None])
    if score_update == "max":
        s_ij = torch.max(s_i, s_j)
    elif score_update == "sum":
        s_ij = s_i.add_(s_j)
    else:
        raise ValueError(f"Invalid args<score_update: {score_update}>")


    # Update the tensors in place
    key_states_l.scatter_(-2, mrg_j_exp, k_ij)
    value_states_l.scatter_(-2, mrg_j_exp, v_ij)
    pos_weight_l.scatter_(-1, mrg_j[..., None], w_ij)
    pos_score_l.scatter_(-1, mrg_j[..., None], s_ij)

    # print(cache_kwargs, mrg_i_exp.shape, key_states.shape, key_states_l.shape)
    #  'cache_budget': 20, 'metric': 'dot_product', 'cache_tail': 2, 'cache_top': 9} torch.Size([1, 32, 1, 128]) torch.Size([1, 32, 0, 128]) torch.Size([1, 32, 8, 128]
    key_states_l.scatter_(-2, mrg_i_exp, key_states)
    value_states_l.scatter_(-2, mrg_i_exp, value_states)
    pos_weight_l.scatter_(-1, mrg_i[..., None], w_n)
    pos_score_l.scatter_(-1, mrg_i[..., None], s_n)

    return key_states_l, value_states_l, pos_weight_l, pos_score_l


def cache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    **cache_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

    Parameters:
        key_states (`torch.Tensor`):
            The new key states to cache.
        value_states (`torch.Tensor`):
            The new value states to cache.
        layer_idx (`int`):
            The index of the layer to cache the states for.
        cache_kwargs (`Dict[str, Any]`, `optional`):
            Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

    Return:
        A tuple containing the updated key and value states.
    """
    
    # :ADD cache
    if "pos_weight" not in dir(self):
        self.pos_weight = []
    if "pos_score" not in dir(self): #  
        self.pos_score = []
    for _ in range(len(self.key_cache), layer_idx + 1):
        self.key_cache.append([])
        self.value_cache.append([])
        self.pos_weight.append([])
        self.pos_score.append([])
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += key_states.shape[-2]

    # arg parse
    init_f = cache_kwargs.pop("init_f")
    cache_budget = cache_kwargs["cache_budget"]
    cache_tail = cache_kwargs["cache_tail"] - 1 # valid!
    scores = cache_kwargs["scores"]  # B, H, Tk
    shrink_factor = cache_kwargs["shrink_factor"]

    if init_f:
        key_states_cat, value_states_cat, pos_weight_cat, scores_cat = cache_init(
                key_states, value_states, **cache_kwargs
            )
        self.key_cache[layer_idx] = key_states_cat
        self.value_cache[layer_idx] = value_states_cat
        self.pos_weight[layer_idx] = pos_weight_cat
        self.pos_score[layer_idx] = scores_cat
        self.tail_ind = -1
    else:
        assert cache_tail >= 1
        # Update pos_score_l in place
        if shrink_factor < 1:
            self.pos_score[layer_idx] *= shrink_factor
        self.pos_score[layer_idx].add_(scores[..., :-1])
        bzs, h, t, d = self.key_cache[layer_idx].size()
        if t < cache_budget:
            self.key_cache[layer_idx] = torch.concat([self.key_cache[layer_idx], key_states], axis=-2)
            self.value_cache[layer_idx] = torch.concat([self.value_cache[layer_idx], value_states], axis=-2)
            self.pos_weight[layer_idx] = torch.concat([self.pos_weight[layer_idx], torch.ones_like(scores[..., -1:])], axis=-1)
            self.pos_score[layer_idx] = torch.concat([self.pos_score[layer_idx], scores[..., -1:]], axis=-1)
            return

        if layer_idx == 0:
            self.tail_ind = (self.tail_ind + 1) % cache_tail
        tail_ind = self.tail_ind
        dyn_ind = cache_budget - cache_tail
        tail_ind += dyn_ind
        k_n, v_n = self.key_cache[layer_idx][..., tail_ind: tail_ind+1, :], self.value_cache[layer_idx][..., tail_ind: tail_ind+1, :]
        w_n, s_n = self.pos_weight[layer_idx][..., tail_ind: tail_ind+1], self.pos_score[layer_idx][..., tail_ind: tail_ind+1]
        
        cache_merge(k_n, v_n, w_n, s_n, self.key_cache[layer_idx][..., :dyn_ind, :], self.value_cache[layer_idx][..., :dyn_ind, :], 
            self.pos_weight[layer_idx][..., :dyn_ind], self.pos_score[layer_idx][..., :dyn_ind], **cache_kwargs
        )

        self.key_cache[layer_idx][..., tail_ind, :] = key_states.squeeze(-2)
        self.value_cache[layer_idx][..., tail_ind, :] = value_states.squeeze(-2)
        self.pos_weight[layer_idx][..., tail_ind] = 1
        self.pos_score[layer_idx][..., tail_ind] = scores[..., -1]
    # if not layer_idx:
        # print(f"scores: \n{cache_kwargs["scores"] [0, 0]}")
        # print(f"{self.pos_score[layer_idx][0, 0]}")
    # #     # print(key_states[0, 0, :, 0])
    # #     # print(key_states_cat[0, 0, :, 0])
    # #     # print(value_states[0, 0, :, 0])
    # #     # print(value_states_cat[0, 0, :, 0])
        # print(f"pos_weight_cat: {self.pos_weight[layer_idx][0, 0]}")
        # print(f"self.tail_ind: {self.tail_ind}")
    return
    

def cache_args_parse(kwargs, q_len):
    cache_budget = kwargs.get("cache_budget", 0.2)
    cache_dense = kwargs.get("cache_dense", 1)
    cache_tail = kwargs.get("cache_tail", 0.1)
    scale_factor = kwargs.setdefault("scale_factor", 1)
    shrink_factor = kwargs.setdefault("shrink_factor", 0.98)
    window_size = kwargs.setdefault("window_size", 8)
    window_pool = kwargs.setdefault("window_pool", "maxpool")
    kernel_size = kwargs.setdefault("kernel_size", 5)
    kwargs.setdefault("metric", "dot_product")
    kwargs.setdefault("score_update", "max")

    # cache_budget
    if 0 < cache_budget < 1:
        cache_budget = int(q_len * cache_budget)
    elif cache_budget >= 1:
        cache_budget = int(cache_budget)
    else:
        raise ValueError(f"invalid arg value: cache_budget=<{cache_budget}>")
    
    # cache_tail
    if 0 < cache_tail < 1:
        cache_tail = int(cache_tail * cache_budget)
    elif cache_tail >= 1:
        cache_tail = int(cache_tail)
    else:
        raise ValueError(f"invalid arg value: cache_tail=<{cache_tail}>")
    cache_tail = max(2, cache_tail)
    assert cache_tail < cache_budget
    
    # cache_dense
    if 0 < cache_dense < 1:
        cache_dense = int((cache_budget - cache_tail) * cache_dense)
    elif cache_dense >= 1:
        cache_dense = int(cache_dense)
    else:
        raise ValueError(f"invalid arg value: cache_dense=<{cache_dense}>")
    cache_dense = max(1, cache_dense)

    # 
    assert cache_budget > cache_tail + cache_dense
    kwargs.update(cache_budget=cache_budget, cache_tail=cache_tail, cache_dense=cache_dense)


def llama_sdpa_attn_forward_(
    self,
    hidden_states: torch.Tensor,
    attention_mask:  torch.Tensor = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
    ) -> Tuple[torch.Tensor,  torch.Tensor, Optional[Tuple[torch.Tensor]]]:
    # kwargs parse
    calss_name = self.__class__.__name__
    if calss_name == "FalconAttention":
        past_key_value = kwargs.get("layer_past")
        num_key_value_groups = 1
    else:
        num_key_value_groups = self.num_key_value_groups
    # args init
    cum_len = position_ids[:, -1].max() + 1
    bsz, q_len, _ = hidden_states.size()

    init_f = True if q_len > 1 else False
    if init_f:
        cache_args_parse(kwargs, q_len)
        self.forward_kwargs = kwargs.copy()
    else:
        kwargs.update(self.forward_kwargs)
    layer_idx = self.layer_idx
    kwargs.update(layer_idx=layer_idx, init_f=init_f)
    cache_budget = kwargs["cache_budget"]
    window_size = kwargs["window_size"]
    window_pool = kwargs["window_pool"]
    kernel_size = kwargs["kernel_size"]
    
    if output_attentions:
        raise NotImplemented(f"output_attentions should be set to False!")
    
    if calss_name == "FalconAttention":
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        batch_size, query_length, _, _ = query_layer.shape
        query_states = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
        key_states = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        value_states = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    bsz, n_h, t_k, d_k = key_states.size()
    bsz, n_head, q_len, n_dim = query_states.size()
    # if bsz > 1:
    #     raise NotImplementedError
    
    assert past_key_value is not None

    # pos_weight = torch.ones_like(position_ids, device=position_ids.device).type(query_states.dtype)[:, None].repeat(1, n_h, 1)
    # print(f"{init_f}: self.num_key_value_groups: {self.num_key_value_groups}", query_states.size(), key_states.size(), value_states.size(), )
    if init_f:
        key_states_ = key_states
        value_states_ = value_states
        pos_weight_ = None

    else:
        key_states_ = torch.concat([past_key_value.key_cache[layer_idx], key_states], dim=-2)
        value_states_ = torch.concat([past_key_value.value_cache[layer_idx], value_states], dim=-2)
        pos_weight_ = torch.concat([past_key_value.pos_weight[layer_idx], torch.ones_like(key_states[..., 0])], dim=-1)

    key_states_ = repeat_kv(key_states_, num_key_value_groups)
    value_states_ = repeat_kv(value_states_, num_key_value_groups)
    pos_weight_ = None if pos_weight_ is None else repeat_kv(pos_weight_, num_key_value_groups) 
        
    scale_factor = kwargs["scale_factor"]
    shrink_factor = kwargs["shrink_factor"]
    attn_output, scores = score_scaled_dot_product_attention(
        query_states,
        key_states_,
        value_states_,
        attn_mask=None,
        pos_weight=pos_weight_,
        scale_factor=scale_factor,
        shrink_factor=shrink_factor,
        init_f=init_f,
        window_size=window_size,
        window_pool=window_pool,
        kernel_size=kernel_size,
    )
    # modified to adapt falcon
    if calss_name == "FalconAttention":
        scores = scores.sum(dim=1, keepdim=True)
    scores = de_repeat_kv(scores, num_key_value_groups)
    cache_update(past_key_value, key_states, value_states, scores=scores, **kwargs)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    if calss_name == "FalconAttention":
        attn_output = self.dense(attn_output)
        return attn_output, past_key_value
        
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def args_dec(func, **kws):
    @wraps(func)
    def dec_func(*args, **kwargs):
        kwargs.update(kws)
        return func(*args, **kwargs)
    return dec_func


class AttentionForward:
    token = ACCESS_TOKEN
    device = "cpu"
    num_gpus = -1
    if torch.cuda.is_available():
        device = "cuda"
        num_gpus = torch.cuda.device_count()

    @classmethod
    def model_load(cls, model_name="meta-llama/Llama-2-7b-hf", merge=True, **kws):
        
        # config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left", token=cls.token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=cls.token)
        cls.change_mode(merge, **kws)
        return tokenizer, model
    
    @classmethod
    def change_mode(cls, merge, **kws):
        if merge:
            transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = args_dec(llama_sdpa_attn_forward_, **kws)
            transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = args_dec(llama_sdpa_attn_forward_, **kws)
            transformers.models.falcon.modeling_falcon.FalconAttention.forward = args_dec(llama_sdpa_attn_forward_, **kws)
        else:
            global g_llama_sdpa_attn_forward_orgn, g_mistral_sdpa_attn_forward_orgn, g_falcon_sdpa_attn_forward_orgn
            transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = g_llama_sdpa_attn_forward_orgn
            transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = g_mistral_sdpa_attn_forward_orgn
            transformers.models.falcon.modeling_falcon.FalconAttention.forward = g_falcon_sdpa_attn_forward_orgn



def main():
    batch_prompts = [
        "What is the capital of France?",
        # "Explain the theory of relativity.",
        # "Describe the process of photosynthesis.",
        # "Who is the author of Pride and Prejudice?",
    ]
    output_len = 6

    tokenizer, model = AttentionForward.model_load(merge=False)
    model.eval().to(AttentionForward.device)

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to(AttentionForward.device)
    batch_input_ids = tokenized_prompts.input_ids

    context_length = batch_input_ids.shape[-1]
    output_max_len = context_length + output_len

    output = model.generate(
        **tokenized_prompts,
        output_attentions = False,
        max_new_tokens=output_max_len,
        num_beams=1,
        do_sample=False,
        top_p=None,
        temperature=1.0,
        min_length=context_length+1,
        eos_token_id=[tokenizer.eos_token_id]
    )

    batch_outputs =tokenizer.batch_decode(output[:, context_length:], skip_special_tokens=True)
    for q, a in zip(batch_prompts, batch_outputs):
        print("-" * 20, f"Q:\n\t{q}\nA:\n\t{a}", sep="\n")
    
    AttentionForward.change_mode(merge=True, cache_budget=18)
    output = model.generate(
        **tokenized_prompts,
        output_attentions = False,
        max_new_tokens=output_max_len,
        num_beams=1,
        do_sample=False,
        top_p=None,
        temperature=1.0,
        min_length=context_length+1,
        eos_token_id=[tokenizer.eos_token_id]
    )

    batch_outputs =tokenizer.batch_decode(output[:, context_length:], skip_special_tokens=True)
    for q, a in zip(batch_prompts, batch_outputs):
        print("-" * 20, f"Q:\n\t{q}\nA:\n\t{a}", sep="\n")


if __name__ == "__main__":
    main()

