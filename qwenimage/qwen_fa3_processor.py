"""
Paired with a good language model. Thanks!

FA3 is currently broken on Blackwell (sm_100) GPUs; this module detects that
at import time and falls back to PyTorch scaled-dot-product attention (SDPA)
automatically.  The public class name / call signature are unchanged.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen


# ---------------------------------------------------------------------------
# FA3 availability check
# ---------------------------------------------------------------------------

def _is_blackwell() -> bool:
    """Return True when the current default CUDA device is an sm_100 (Blackwell) GPU."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    # Blackwell → compute capability 10.x  (sm_100)
    return cap[0] >= 10


_fa3_available: bool = False
_fa3_unavailable_reason: str = ""
_flash_attn_func = None

if _is_blackwell():
    _fa3_unavailable_reason = (
        "FlashAttention-3 is not yet supported on Blackwell (sm_100) GPUs. "
        "Falling back to scaled-dot-product attention (SDPA)."
    )
else:
    try:
        from kernels import get_kernel
        _k = get_kernel("kernels-community/vllm-flash-attn3")
        _flash_attn_func = _k.flash_attn_func
        _fa3_available = True
    except Exception as e:
        _fa3_unavailable_reason = (
            "FlashAttention-3 via Hugging Face `kernels` is unavailable. "
            f"Tried `get_kernel('kernels-community/vllm-flash-attn3')` and failed with:\n{e}\n"
            "Falling back to scaled-dot-product attention (SDPA)."
        )


# ---------------------------------------------------------------------------
# FA3 custom op (registered only when the kernel loaded successfully)
# ---------------------------------------------------------------------------

if _fa3_available:
    @torch.library.custom_op("flash::flash_attn_func", mutates_args=())
    def flash_attn_func(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        # _flash_attn_func returns (output, softmax_lse); we only need output.
        output, _lse = _flash_attn_func(q, k, v, causal=causal)
        return output

    @flash_attn_func.register_fake
    def _flash_attn_func_fake(q, k, v, causal=False):
        # output shape mirrors q: (batch, seq_len, num_heads, head_dim)
        return torch.empty_like(q).contiguous()

else:
    # Provide a stub so call-sites that import the symbol don't break at
    # module load; the processor will route around it at runtime.
    def flash_attn_func(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        raise RuntimeError(_fa3_unavailable_reason)


# ---------------------------------------------------------------------------
# SDPA fallback helper
# ---------------------------------------------------------------------------

def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Scaled dot-product attention using torch.nn.functional.scaled_dot_product_attention.

    Input / output layout: (B, S, H, D_h)  — same as the FA3 kernel.
    """
    # SDPA expects (B, H, S, D_h)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # Back to (B, S, H, D_h)
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Attention processor
# ---------------------------------------------------------------------------

class QwenDoubleStreamAttnProcessorFA3:
    """
    Attention processor for the Qwen double-stream architecture.

    Preferred backend: vLLM FlashAttention-3 via Hugging Face ``kernels``.
    Automatic fallback: PyTorch ``scaled_dot_product_attention`` (SDPA) when
    FA3 is unavailable — e.g. on Blackwell (sm_100) GPUs where FA3 is not yet
    supported, or when the ``kernels`` package is absent.

    Notes / limitations
    -------------------
    - Arbitrary attention masks are not supported on the FA3 path.  Pass
      ``attention_mask=None`` (the default) to stay on the fast path.
    - On the SDPA path, ``attention_mask`` is likewise ignored; add explicit
      support here if you need it.
    - ``encoder_hidden_states`` (text stream) is required.
    """

    _attention_backend: str  # set in __init__ after capability detection

    def __init__(self):
        if _fa3_available:
            self._attention_backend = "fa3"
        else:
            import warnings
            warnings.warn(
                f"QwenDoubleStreamAttnProcessorFA3: {_fa3_unavailable_reason}",
                stacklevel=2,
            )
            self._attention_backend = "sdpa"

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """Dispatch to FA3 or SDPA depending on what is available."""
        if self._attention_backend == "fa3":
            return flash_attn_func(q, k, v, causal=causal)
        return _sdpa_attention(q, k, v, causal=causal)

    @torch.no_grad()
    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,                          # (B, S_img, D_model)
        encoder_hidden_states: torch.FloatTensor = None,           # (B, S_txt, D_model)
        encoder_hidden_states_mask: torch.FloatTensor = None,      # unused
        attention_mask: Optional[torch.FloatTensor] = None,        # unsupported on FA3 path
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        if encoder_hidden_states is None:
            raise ValueError(
                "QwenDoubleStreamAttnProcessorFA3 requires encoder_hidden_states (text stream)."
            )
        if attention_mask is not None and self._attention_backend == "fa3":
            raise NotImplementedError(
                "attention_mask is not supported on the FA3 path. "
                "Either drop the mask or let the processor fall back to SDPA."
            )

        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]

        # ---- QKV projections ----
        img_q = attn.to_q(hidden_states)
        img_k = attn.to_k(hidden_states)
        img_v = attn.to_v(hidden_states)

        txt_q = attn.add_q_proj(encoder_hidden_states)
        txt_k = attn.add_k_proj(encoder_hidden_states)
        txt_v = attn.add_v_proj(encoder_hidden_states)

        # ---- Reshape to (B, S, H, D_h) ----
        H = attn.heads
        img_q = img_q.unflatten(-1, (H, -1))
        img_k = img_k.unflatten(-1, (H, -1))
        img_v = img_v.unflatten(-1, (H, -1))

        txt_q = txt_q.unflatten(-1, (H, -1))
        txt_k = txt_k.unflatten(-1, (H, -1))
        txt_v = txt_v.unflatten(-1, (H, -1))

        # ---- Q/K normalization ----
        if getattr(attn, "norm_q", None) is not None:
            img_q = attn.norm_q(img_q)
        if getattr(attn, "norm_k", None) is not None:
            img_k = attn.norm_k(img_k)
        if getattr(attn, "norm_added_q", None) is not None:
            txt_q = attn.norm_added_q(txt_q)
        if getattr(attn, "norm_added_k", None) is not None:
            txt_k = attn.norm_added_k(txt_k)

        # ---- RoPE (Qwen variant) ----
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)

        # ---- Joint attention over [text, image] along sequence axis ----
        q = torch.cat([txt_q, img_q], dim=1)  # (B, S_txt + S_img, H, D_h)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)

        out = self._attend(q, k, v, causal=False)  # (B, S_total, H, D_h)

        # ---- Back to (B, S, D_model) ----
        out = out.flatten(2, 3).to(q.dtype)

        # ---- Split text / image segments ----
        txt_attn_out = out[:, :S_txt, :]
        img_attn_out = out[:, S_txt:, :]

        # ---- Output projections ----
        img_attn_out = attn.to_out[0](img_attn_out)
        if len(attn.to_out) > 1:
            img_attn_out = attn.to_out[1](img_attn_out)  # dropout if present

        txt_attn_out = attn.to_add_out(txt_attn_out)

        return img_attn_out, txt_attn_out
