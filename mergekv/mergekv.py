from typing import Optional, Tuple
from dataclasses import dataclass, field
import os
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention
from dotenv import load_dotenv

from transformers.utils import logging

# Configure logger
logger = logging.get_logger(__name__)

# Load environment variables
load_dotenv()
ACCESS_TOKEN=os.getenv("ACCESS_TOKEN")


@dataclass
class AttForwardArgs:
    """Arguments for attention forward computation and cache management."""
    # Cache configuration
    cache_budget: float = 0.2
    cache_dense: float = 0.02
    cache_tail: float = 0.5

    # Attention parameters
    scale_factor: float = 1.0
    shrink_factor: float = 0.98
    window_size: int = 8
    window_pool: Optional[str] = None
    kernel_size: int = 5

    # Other parameters
    out_state: int = 0
    metric: str = "l2"
    score_update: str = "sum"

    # Computed values (set during initialization)
    cache_budget_abs: int = field(init=False, default=0)
    cache_tail_abs: int = field(init=False, default=0)
    cache_dense_abs: int = field(init=False, default=0)

    def initialize_for_sequence(self, seq_len: int) -> None:
        """Initialize absolute cache sizes based on sequence length."""
        # Cache budget
        if 0 < self.cache_budget < 1:
            self.cache_budget_abs = int(seq_len * self.cache_budget)
        elif self.cache_budget >= 1:
            self.cache_budget_abs = int(self.cache_budget)
        else:
            raise ValueError(f"invalid arg value: cache_budget=<{self.cache_budget}>")

        # Cache tail
        if 0 < self.cache_tail < 1:
            self.cache_tail_abs = int(self.cache_tail * self.cache_budget_abs)
        elif self.cache_tail >= 1:
            self.cache_tail_abs = int(self.cache_tail)
        else:
            raise ValueError(f"invalid arg value: cache_tail=<{self.cache_tail}>")
        self.cache_tail_abs = max(2, self.cache_tail_abs)

        # Cache dense
        if self.cache_dense == 0:
            self.cache_dense_abs = 0
        elif 0 < self.cache_dense < 1:
            self.cache_dense_abs = int((self.cache_budget_abs - self.cache_tail_abs) * self.cache_dense)
            self.cache_dense_abs = max(1, self.cache_dense_abs)
        elif self.cache_dense >= 1:
            self.cache_dense_abs = int(self.cache_dense)
        else:
            raise ValueError(f"invalid arg value: cache_dense=<{self.cache_dense}>")

        # Validation
        assert self.cache_tail_abs < self.cache_budget_abs
        assert self.cache_budget_abs > self.cache_tail_abs + self.cache_dense_abs


global g_llama_sdpa_attn_forward_orgn, g_mistral_sdpa_attn_forward_orgn, g_falcon_sdpa_attn_forward_orgn
g_llama_sdpa_attn_forward_orgn = transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
g_mistral_sdpa_attn_forward_orgn = transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward
g_falcon_sdpa_attn_forward_orgn = transformers.models.falcon.modeling_falcon.FalconAttention.forward
global g_qwen_sdpa_attn_forward_orgn
g_qwen_sdpa_attn_forward_orgn = Qwen2SdpaAttention.forward


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value tensors for multi-head attention."""
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
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        scale_factor: Optional[float] = None,
        shrink_factor: Optional[float] = None,
        init_f: bool = False,
        window_size: Optional[int] = None,
        window_pool: Optional[str] = None,
        kernel_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def sub_sdpa(
            t_bias: int,
            sub_query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            sub_attn_mask: Optional[torch.Tensor] = None,
            pos_weight: Optional[torch.Tensor] = None,
            scale_factor: Optional[float] = None,
            shrink_factor: Optional[float] = None,
            need_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = sub_query.dtype
        _, _, t_k, d_k = key.size()
        _, _, t_q, _ = sub_query.size()
        scale = 1.0 / torch.sqrt(torch.tensor(d_k, dtype=dtype))
        
        if sub_attn_mask is None and  need_mask:
            mask_dtype = torch.float32
            sub_attn_mask = torch.triu(torch.full((t_q, t_k), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=key.device), diagonal=t_k - t_bias + 1)
            sub_attn_mask = sub_attn_mask[None, None, ...]
        
        with torch.amp.autocast("cuda", dtype=torch.float32):
            scores = torch.matmul(sub_query, key.transpose(-2, -1))
            scores = scores * scale
            if sub_attn_mask is not None:
                scores = scores + sub_attn_mask

            # softmax
            if pos_weight is not None:
                scores += scale_factor * torch.log(pos_weight)[..., None, :]
            attn_weights = F.softmax(scores, dim=-1) # , dtype=value.dtype

            output = torch.matmul(attn_weights, value)
        output = output.to(value.dtype)
        if shrink_factor < 1 and t_q > 1:
            shrink_f = torch.pow(shrink_factor, torch.arange(t_q, device=attn_weights.device))
            shrink_f_flipped = torch.flip(shrink_f, dims=[0])
            score = (attn_weights * shrink_f_flipped[None, None, :, None]).sum(dim=-2)
        else:
            score = attn_weights.sum(dim=-2)
        return output, score
    
    bsz, n_head, t_k, _ = key.size()
    _, _, t_q, _ = query.size()

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
        bsz, n_head, t_q, _ = query.size()
        attn_mask = attn_mask[:, :, -window_size:, :] if attn_mask is not None else None
    need_mask =  t_q > 1
    sub_len = t_q
    shrink_factor_tensor = torch.tensor(shrink_factor or 1.0, dtype=key.dtype, device=key.device)
    while 1:
        try:
            score = torch.zeros(bsz, n_head, t_k, dtype=key.dtype, device=key.device)
            output = []
            for start_ind in range(0, t_q, sub_len):
                sub_query = query[:, :, start_ind : start_ind + sub_len]
                sub_attn_mask = attn_mask[:, :, start_ind : start_ind + sub_len] if attn_mask else None
                output_sub, score_ = sub_sdpa(t_q - start_ind, sub_query, key, value, sub_attn_mask, pos_weight, scale_factor, shrink_factor, need_mask=need_mask)
                output.append(output_sub)
                score *= torch.pow(shrink_factor_tensor, sub_query.size(2))
                score += score_
            else:
                output = torch.cat(output, dim=2)
                break
        except torch.OutOfMemoryError as e:
            if sub_len<= 10:
                raise e
            sub_len = (sub_len + 3) // 4
    if init_f:
        assert attn_output is not None
        output = attn_output
        if window_pool == 'avgpool' and kernel_size is not None:
            score = F.avg_pool1d(score, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
        elif window_pool == 'maxpool' and kernel_size is not None:
            score = F.max_pool1d(score, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    return output, score


def topk_split_new(E, S, k, tail_max):
    """
    Split data tensor E into head A and tail B based on score tensor S.
    Head A contains the top-k highest scoring elements.
    Tail B contains the highest scoring tail_max elements from the remaining elements,
    preserving their original relative order.

    Args:
        E (Tensor): Input data tensor with shape [b, t, d].
        S (Tensor): Score tensor with shape [b, t].
        k (int): Size of head A.
        tail_max (int): Size of tail B. If remaining elements < tail_max, takes all remaining.
                        If 0, returns no tail elements.

    Returns:
        A (Tensor): Head tensor with shape [b, k, d].
        B (Tensor): Tail tensor with shape [b, actual_tail_max, d].
        topk_values (Tensor): Scores corresponding to elements in A, shape [b, k].
        tail_values (Tensor): Scores corresponding to elements in B, shape [b, actual_tail_max].
    """
    b, t, d = E.shape

    # Step 1: Sort scores for entire sequence, get indices sorted by score (high to low)
    _, sorted_indices = torch.sort(S, dim=1, descending=True)

    # Step 2: Head A indices: top k elements
    topk_indices = sorted_indices[:, :k]

    # Step 3: Gather head A
    A = torch.gather(E, 1, topk_indices.unsqueeze(-1).expand(-1, -1, d))
    topk_values = torch.gather(S, 1, topk_indices)

    # Step 4: Handle tail B
    if tail_max == 0:
        # If tail_max == 0, return empty tensors
        B = torch.empty(b, 0, d, dtype=E.dtype, device=E.device)
        tail_values = torch.empty(b, 0, dtype=S.dtype, device=S.device)
    else:
        # Tail B indices: next tail_max elements after head
        # Handle boundary case where t-k < tail_max
        num_remaining = t - k
        actual_tail_max = min(tail_max, num_remaining)

        tail_indices_sorted_by_score = sorted_indices[:, k : k + actual_tail_max]

        # Step 5: Restore original relative order of tail indices
        # tail_indices_sorted_by_score is currently sorted by score, e.g., [8, 3, 5]
        # To maintain original order, sort these index values to get [3, 5, 8]
        # This way gather will take elements in original sequence order
        final_tail_indices, _ = torch.sort(tail_indices_sorted_by_score, dim=1)

        # Step 6: Gather tail B (using final_tail_indices with restored original order)
        B = torch.gather(E, 1, final_tail_indices.unsqueeze(-1).expand(-1, -1, d))
        tail_values = torch.gather(S, 1, final_tail_indices)

    return A, B, topk_values, tail_values


def cache_init(
        key_states, value_states, scores, args: AttForwardArgs
    ):
    cache_budget = args.cache_budget_abs
    cache_tail = args.cache_tail_abs
    cache_dense = args.cache_dense_abs
    score_update = args.score_update
    cache_top_size = int((cache_budget - cache_tail) - cache_dense)

    bsz, n_head, t_k, d_k = key_states.size()
    if t_k <= cache_budget:
        pos_weight = torch.ones_like(scores)
        return key_states, value_states, pos_weight, scores

    key_li, value_li, score_li, weight_li = [], [], [], []
    for key, value, score in zip(key_states, value_states, scores):
        assert t_k >= cache_budget
        kv_status = torch.cat([key, value], dim=-1)
        kv_status_ = kv_status[..., :t_k-cache_tail, :]
        score_ = score[..., :t_k-cache_tail]
        
        cache_top, cache_residual, score_top, score_residual = topk_split_new(
            kv_status_.view(n_head, t_k-cache_tail, -1),
            score_.view(n_head, t_k-cache_tail),
            k=cache_top_size,
            tail_max=int(cache_dense * 10)
        )

        # Handle cache_dense == 0: drop all residual values
        if cache_dense == 0:
            # No dense region, only keep top and tail
            kv_status_cat = torch.cat([cache_top, kv_status[..., t_k-cache_tail:, :]], dim=-2)
            score_cat = torch.cat([score_top, score[..., t_k-cache_tail:]], dim=-1)
            weight_cat = torch.cat([torch.ones_like(score_top), torch.ones_like(score[..., t_k-cache_tail:])], dim=-1)
        else:
            merge_size, tail_add = divmod(cache_residual.size(-2), cache_dense)

            # Handle uneven division: first tail_add groups get merge_size+1 elements
            sep_id = tail_add * (merge_size + 1)
            cache_residual_m = torch.cat([
                cache_residual[..., :sep_id, :].view(n_head, tail_add, merge_size + 1, 2 * d_k).mean(dim=-2),
                cache_residual[..., sep_id:, :].view(n_head, cache_dense - tail_add, merge_size, 2 * d_k).mean(dim=-2),
            ], dim=-2)
            if score_update == "max":
                socre_m = torch.cat([
                    score_residual[..., :sep_id].view(n_head, tail_add, merge_size + 1).max(dim=-1)[0],
                    score_residual[..., sep_id:].view(n_head, cache_dense - tail_add, merge_size).max(dim=-1)[0],
                ], dim=-1)
            elif score_update == "sum":
                socre_m = torch.cat([
                    score_residual[..., :sep_id].view(n_head, tail_add, merge_size + 1).sum(dim=-1),
                    score_residual[..., sep_id:].view(n_head, cache_dense - tail_add, merge_size).sum(dim=-1),
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


def cache_merge(key_states, value_states, w_n, s_n, key_states_l, value_states_l, pos_weight_l, pos_score_l, args: AttForwardArgs):
    metric = args.metric
    score_update = args.score_update
    cache_dense = args.cache_dense_abs

    bsz, n_head, cache_t, d_k = key_states_l.size()

    # Handle cache_dense == 0: simply drop the incoming values
    if cache_dense == 0:
        # No dense region to merge with, just find the minimum score in the entire cache
        # and replace it with the new token
        _, mrg_i = torch.min(pos_score_l, dim=-1)
        mrg_i_exp = mrg_i[..., None, None].expand(-1, -1, -1, d_k)

        # Replace the lowest-scoring token with the new token
        key_states_l.scatter_(-2, mrg_i_exp, key_states)
        value_states_l.scatter_(-2, mrg_i_exp, value_states)
        pos_weight_l.scatter_(-1, mrg_i[..., None], w_n)
        pos_score_l.scatter_(-1, mrg_i[..., None], s_n)

        return key_states_l, value_states_l, pos_weight_l, pos_score_l

    # Find the minimum index for pos_score_l (exclude dense region)
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

    key_states_l.scatter_(-2, mrg_i_exp, key_states)
    value_states_l.scatter_(-2, mrg_i_exp, value_states)
    pos_weight_l.scatter_(-1, mrg_i[..., None], w_n)
    pos_score_l.scatter_(-1, mrg_i[..., None], s_n)

    return key_states_l, value_states_l, pos_weight_l, pos_score_l


def cache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scores: torch.Tensor,
    args: AttForwardArgs,
    layer_idx: int,
    init_f: bool,
) -> Cache:
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

    # Parse arguments
    cache_budget = args.cache_budget_abs
    cache_tail = args.cache_tail_abs - 1
    shrink_factor = args.shrink_factor

    if init_f:
        key_states_cat, value_states_cat, pos_weight_cat, scores_cat = cache_init(
                key_states, value_states, scores=scores, args=args
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
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            self.pos_weight[layer_idx] = torch.cat([self.pos_weight[layer_idx], torch.ones_like(scores[..., -1:])], dim=-1)
            self.pos_score[layer_idx] = torch.cat([self.pos_score[layer_idx], scores[..., -1:]], dim=-1)
            return self

        if layer_idx == 0:
            self.tail_ind = (self.tail_ind + 1) % cache_tail
        tail_ind = self.tail_ind
        dyn_ind = cache_budget - cache_tail
        tail_ind += dyn_ind
        k_n, v_n = self.key_cache[layer_idx][..., tail_ind: tail_ind+1, :], self.value_cache[layer_idx][..., tail_ind: tail_ind+1, :]
        w_n, s_n = self.pos_weight[layer_idx][..., tail_ind: tail_ind+1], self.pos_score[layer_idx][..., tail_ind: tail_ind+1]
        
        cache_merge(k_n, v_n, w_n, s_n, self.key_cache[layer_idx][..., :dyn_ind, :], self.value_cache[layer_idx][..., :dyn_ind, :],
            self.pos_weight[layer_idx][..., :dyn_ind], self.pos_score[layer_idx][..., :dyn_ind], args=args
        )

        self.key_cache[layer_idx][..., tail_ind, :] = key_states.squeeze(-2)
        self.value_cache[layer_idx][..., tail_ind, :] = value_states.squeeze(-2)
        self.pos_weight[layer_idx][..., tail_ind] = 1
        self.pos_score[layer_idx][..., tail_ind] = scores[..., -1]
    return self
    

def create_att_forward_args(**kwargs) -> AttForwardArgs:
    """Create AttForwardArgs from keyword arguments."""
    # Extract dataclass fields
    dataclass_fields = {
        'cache_budget': kwargs.get('cache_budget', 0.2),
        'cache_dense': kwargs.get('cache_dense', 0.02),
        'cache_tail': kwargs.get('cache_tail', 0.5),
        'scale_factor': kwargs.get('scale_factor', 1.0),
        'shrink_factor': kwargs.get('shrink_factor', 0.98),
        'window_size': kwargs.get('window_size', 8),
        'window_pool': kwargs.get('window_pool'),
        'kernel_size': kwargs.get('kernel_size', 5),
        'out_state': kwargs.get('out_state', 0),
        'metric': kwargs.get('metric', 'l2'),
        'score_update': kwargs.get('score_update', 'sum'),
    }

    return AttForwardArgs(**dataclass_fields)


def sdpa_attn_forward_variant(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    # Parse class-specific parameters
    class_name = self.__class__.__name__
    if class_name == "FalconAttention":
        past_key_value = kwargs.get("layer_past")
        num_key_value_groups = 1
    else:
        num_key_value_groups = self.num_key_value_groups

    # Initialize arguments
    bsz, q_len, _ = hidden_states.size()
    init_f = q_len > 1

    if init_f:
        # Create and initialize args for new sequence
        self.forward_args = create_att_forward_args(**kwargs)
        self.forward_args.initialize_for_sequence(q_len)
    else:
        # Use existing args for incremental generation
        if not hasattr(self, 'forward_args'):
            raise RuntimeError("forward_args not initialized. This should not happen.")

    layer_idx = self.layer_idx
    args = self.forward_args
    
    if output_attentions:
        raise NotImplementedError("output_attentions should be set to False!")
    
    if class_name == "FalconAttention":
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

    _, _, t_k, d_k = key_states.size()
    bsz, n_head, q_len, _ = query_states.size()
    
    assert past_key_value is not None

    if init_f:
        key_states_ = key_states
        value_states_ = value_states
        pos_weight_ = None

    else:
        key_states_ = torch.cat([past_key_value.key_cache[layer_idx], key_states], dim=-2)
        value_states_ = torch.cat([past_key_value.value_cache[layer_idx], value_states], dim=-2)
        pos_weight_ = torch.cat([past_key_value.pos_weight[layer_idx], torch.ones_like(key_states[..., 0])], dim=-1)

    key_states_ = repeat_kv(key_states_, num_key_value_groups)
    value_states_ = repeat_kv(value_states_, num_key_value_groups)
    pos_weight_ = None if pos_weight_ is None else repeat_kv(pos_weight_, num_key_value_groups) 
        
    attn_output, scores = score_scaled_dot_product_attention(
        query_states,
        key_states_,
        value_states_,
        attn_mask=None,
        pos_weight=pos_weight_,
        scale_factor=args.scale_factor,
        shrink_factor=args.shrink_factor,
        init_f=init_f,
        window_size=args.window_size,
        window_pool=args.window_pool,
        kernel_size=args.kernel_size,
    )
    # Modified to adapt falcon
    if class_name == "FalconAttention":
        scores = scores.sum(dim=1, keepdim=True)
    scores = de_repeat_kv(scores, num_key_value_groups)
    past_key_value = cache_update(past_key_value, key_states, value_states, scores=scores, args=args, layer_idx=layer_idx, init_f=init_f)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    if class_name == "FalconAttention":
        attn_output = self.dense(attn_output)
        return attn_output, past_key_value
        
    attn_output = self.o_proj(attn_output)
    # Bias measurement
    if args.out_state:
        key_states_2 = past_key_value.key_cache[layer_idx]
        value_states_2 = past_key_value.value_cache[layer_idx]
        pos_weight_2 = past_key_value.pos_weight[layer_idx]
        
        key_states_2 = repeat_kv(key_states_2, num_key_value_groups)
        value_states_2 = repeat_kv(value_states_2, num_key_value_groups)
        pos_weight_2 = None if pos_weight_2 is None else repeat_kv(pos_weight_2, num_key_value_groups) 
        # pos_weight_2[:, :, 0] = 0.0001
        attn_output_0, scores = score_scaled_dot_product_attention(
            query_states[:, :, -1:, :],
            key_states_2,
            value_states_2,
            attn_mask=None,
            pos_weight=pos_weight_2,
            scale_factor=0,
            shrink_factor=args.shrink_factor,
            init_f=False,
            window_size=args.window_size,
            window_pool=args.window_pool,
            kernel_size=args.kernel_size,
        )
        attn_output_1, scores = score_scaled_dot_product_attention(
            query_states[:, :, -1:, :],
            key_states_2,
            value_states_2,
            attn_mask=None,
            pos_weight=pos_weight_2,
            scale_factor=1,
            shrink_factor=args.shrink_factor,
            init_f=False,
            window_size=args.window_size,
            window_pool=args.window_pool,
            kernel_size=args.kernel_size,
        )
        
        attn_output_ = torch.cat([attn_output_0, attn_output_1], dim=0).transpose(1, 2).contiguous()
        attn_output_ = attn_output_.view(bsz * 2, 1, -1)

        if class_name != "FalconAttention":
            attn_output_ = self.o_proj(attn_output_)
        self.out_state = torch.cat([attn_output[:, -1:], attn_output_], dim=0)
    return attn_output, None, past_key_value




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
        """Change attention mode between original and MergeKV variant.

        Args:
            merge: If True, use MergeKV attention. If False, use original attention.
            **kws: Additional arguments for AttForwardArgs configuration.
        """
        if merge:
            # Create a wrapper that passes the configuration to the variant
            def make_forward_wrapper(**config_kws):
                def forward_wrapper(self, *args, **kwargs):
                    kwargs.update(config_kws)
                    return sdpa_attn_forward_variant(self, *args, **kwargs)
                return forward_wrapper

            wrapper = make_forward_wrapper(**kws)
            transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = wrapper
            transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = wrapper
            transformers.models.falcon.modeling_falcon.FalconAttention.forward = wrapper
            Qwen2SdpaAttention.forward = wrapper
        else:
            global g_llama_sdpa_attn_forward_orgn, g_mistral_sdpa_attn_forward_orgn, g_falcon_sdpa_attn_forward_orgn
            transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = g_llama_sdpa_attn_forward_orgn
            transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = g_mistral_sdpa_attn_forward_orgn
            transformers.models.falcon.modeling_falcon.FalconAttention.forward = g_falcon_sdpa_attn_forward_orgn
            Qwen2SdpaAttention.forward = g_qwen_sdpa_attn_forward_orgn




def main():
    batch_prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "Describe the process of photosynthesis.",
        "Who is the author of Pride and Prejudice?",
    ]
    output_len = 6

    tokenizer, model = AttentionForward.model_load(merge=False)
    model.eval().to(AttentionForward.device)

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to(AttentionForward.device)
    batch_input_ids = tokenized_prompts.input_ids

    context_length = batch_input_ids.shape[-1]

    output = model.generate(
        **tokenized_prompts,
        output_attentions = False,
        max_new_tokens=output_len,
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
    
    AttentionForward.change_mode(merge=True, cache_budget=10, cache_dense=0)
    output = model.generate(
        **tokenized_prompts,
        output_attentions = False,
        max_new_tokens=output_len,
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

