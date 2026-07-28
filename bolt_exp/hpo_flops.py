"""Training-FLOPs estimate for the BoLT HPO problems.

This module is deliberately written as if it already lived in
``bolt/problems/hpo.py``: :func:`flops` is a pure, batched function of the raw
``X`` tensor with the same signature and semantics as the existing
``LLMTestProblem.cost(X)``, so it can be lifted into the problem classes later
to drive cost-aware BO.

The estimate
------------
The textbook accounting for training a dense transformer is ``6 N D``: ``2 N D``
for the forward pass, and ``4 N D`` for the backward pass, which splits evenly
into input-gradients (``2 N D``) and weight-gradients (``2 N D``). ``N`` counts
matmul parameters, ``D`` counts training tokens.

LoRA fine-tuning changes exactly two things, and those two changes *are* this
formula:

1. **The base weights are frozen.** No weight-gradients are computed for them,
   so the backward pass costs ``2 N D`` instead of ``4 N D``.
2. **Gradients only have to reach the shallowest trainable layer.** With
   ``lora_layers = last_k`` the adapters sit on the top ``k`` blocks, so
   autograd never descends below block ``L - k``: the backward pass traverses
   ``k`` blocks, not ``L``.

Writing ``P_block`` for the matmul parameters in one transformer block,
``P_head = h * V`` for the lm_head, and ``s`` for the effective sequence length::

    N_fwd = L * P_block + P_head      # every block is traversed forward
    N_bwd = k * P_block + P_head      # only the trainable tail is traversed backward
    A     = 4 * s * nh * hd           # QK^T and AV, per token per block

    FLOPs = 2 * D * (N_fwd + N_bwd)  +  A * D * (L + 2 * k)

The attention term ``A`` covers the two weightless batched matmuls in
self-attention (scores and the value-weighted sum); it carries no parameters, so
it cannot appear in ``N``. It contributes ~10% of the total at ``s = 1024``.

What is left out, and why
-------------------------
The LoRA adapters' own forward/backward/weight-gradient FLOPs are omitted: at
``r <= 32`` they are ~0.1% of the total. Normalisations, activations, softmax and
the optimizer step are likewise below the noise floor of this estimate.

The honest consequence is worth stating plainly: **learning rate, dropout,
lora_r and lora_alpha do not change the FLOPs, and neither does batch size at a
fixed token budget.** The cost drivers are model size, training tokens, and how
deep the LoRA adapters go.

Calibration
-----------
The one free constant is the effective sequence length ``s``. The tokenizer's
``max_len`` is 2048 (``bolt-data/bolt_data/common/preprocess.py``) and
sequences are padded/truncated below it; ``s = 1024`` is used.

At that value the formula reproduces the independent estimate for the data
mixture experiment -- Qwen3-4B, all 36 layers, the fixed two-stage 5e6 -> 1e7
token schedule -- as 1.790e17 FLOPs against 1.8e17, i.e. within 0.6%.

Not used: HuggingFace Trainer's recorded ``train/total_flos``
------------------------------------------------------------
The raw HPO runs logged ``train/total_flos``, but that estimator counts every
parameter as trainable and is therefore blind to LoRA: across the 7091 (4B) and
7404 (8B) recorded rows, ``lora_layers``, ``lora_r`` and ``lora_target_modules``
have no measurable effect on it (within-group spread <1%). Its only non-token
driver is per-device batch size, which enters through sequence padding rather
than through compute.
"""

import torch

# Qwen3 architecture constants, from the model config.json files.
QWEN3_4B = {
    "name": "Qwen3-4B",
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}
QWEN3_8B = {
    "name": "Qwen3-8B",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}

# Effective sequence length; the single calibrated constant (see module docstring).
SEQ_LEN = 1024

# Token budget of a full-fidelity HPO run, and the low end of the fidelity range.
FULL_BUDGET_TOKENS = 9.1e6
MIN_BUDGET_TOKENS = 1e5

# lora_layers is searched as an integer in [1, 30] that snaps to one of these
# options; 30 is the proxy for "all", i.e. every transformer block.
LORA_LAYER_OPTIONS = [1, 5, 10, 15, 20, 25, 30]
LORA_LAYERS_ALL = 30

# Data mixture reference point: Qwen3-4B, all layers, 5e6 + 5e6 tokens.
DMO_TOKENS = 1e7
DMO_ANCHOR_FLOPS = 1.8e17

# Which HPO problem the X rows come from.
VARIANTS = ("hpo", "hpo_fd_step", "hpo_fd_model")


def block_matmul_params(arch: dict) -> int:
    """Matmul parameters in one transformer block (attention + MLP projections)."""
    h = arch["hidden_size"]
    nh, nkv, hd = (
        arch["num_attention_heads"],
        arch["num_key_value_heads"],
        arch["head_dim"],
    )
    attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h  # q, k, v, o
    mlp = 3 * h * arch["intermediate_size"]  # gate, up, down
    return attn + mlp


def head_matmul_params(arch: dict) -> int:
    """Matmul parameters in the lm_head."""
    return arch["hidden_size"] * arch["vocab_size"]


def run_flops(
    arch: dict,
    tokens: torch.Tensor | float,
    trainable_layers: torch.Tensor | float,
    seq_len: int = SEQ_LEN,
) -> torch.Tensor:
    """FLOPs of one LoRA fine-tuning run.

    Args:
        arch: One of `QWEN3_4B` / `QWEN3_8B`.
        tokens: Number of training tokens seen, `D`.
        trainable_layers: Number of top blocks carrying LoRA adapters, `k`.
        seq_len: Effective sequence length, `s`.

    Returns:
        Estimated FLOPs, broadcast over `tokens` and `trainable_layers`.
    """
    tokens = torch.as_tensor(tokens, dtype=torch.double)
    k = torch.as_tensor(trainable_layers, dtype=torch.double)

    n_layers = arch["num_hidden_layers"]
    p_block = block_matmul_params(arch)
    p_head = head_matmul_params(arch)
    attn_per_token_block = 4 * seq_len * arch["num_attention_heads"] * arch["head_dim"]

    n_fwd = n_layers * p_block + p_head
    n_bwd = k * p_block + p_head
    matmul = 2 * tokens * (n_fwd + n_bwd)
    attention = attn_per_token_block * tokens * (n_layers + 2 * k)
    return matmul + attention


def decode(X: torch.Tensor, variant: str = "hpo") -> dict:
    """Decode raw HPO `X` rows into the quantities the FLOPs formula needs.

    Args:
        X: `(N, 7)` for `hpo`, `(N, 8)` for the multi-fidelity variants, in the
            problems' own raw bounds.
        variant: One of `VARIANTS`.

    Returns:
        Dict with `arch` (list of arch dicts, one per row), `tokens`,
        `trainable_layers`, `lora_layers` (the snapped raw option) and
        `model_name`.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")

    X = torch.as_tensor(X, dtype=torch.double)
    if X.ndim == 1:
        X = X.unsqueeze(0)

    expected_dim = 7 if variant == "hpo" else 8
    if X.shape[-1] != expected_dim:
        raise ValueError(
            f"variant {variant!r} expects {expected_dim} columns, got {X.shape[-1]}"
        )

    # Column 5 is lora_layers in [1, 30]; snap to the nearest searched option.
    options = torch.tensor(LORA_LAYER_OPTIONS, dtype=torch.double)
    snapped = options[(X[:, 5, None] - options).abs().argmin(dim=-1)]
    n_layers = QWEN3_8B["num_hidden_layers"]
    trainable = torch.where(
        snapped == LORA_LAYERS_ALL, torch.full_like(snapped, n_layers), snapped
    )

    if variant == "hpo_fd_step":
        # Column 7 is fidelity in [0, 1], linear over the token budget.
        fidelity = X[:, 7].clamp(0.0, 1.0)
        tokens = MIN_BUDGET_TOKENS + fidelity * (FULL_BUDGET_TOKENS - MIN_BUDGET_TOKENS)
    else:
        tokens = torch.full((len(X),), FULL_BUDGET_TOKENS, dtype=torch.double)

    if variant == "hpo_fd_model":
        # Column 7 is model size: 0 for Qwen3-4B, 1 for Qwen3-8B.
        is_8b = X[:, 7].round().bool()
    else:
        is_8b = torch.ones(len(X), dtype=torch.bool)

    arch = [QWEN3_8B if b else QWEN3_4B for b in is_8b.tolist()]
    return {
        "arch": arch,
        "model_name": [a["name"] for a in arch],
        "tokens": tokens,
        "trainable_layers": trainable,
        "lora_layers": snapped,
    }


def flops(
    X: torch.Tensor, variant: str = "hpo", seq_len: int = SEQ_LEN
) -> torch.Tensor:
    """Estimated training FLOPs for each raw HPO query.

    Args:
        X: `(N, 7)` for `hpo`, `(N, 8)` for the multi-fidelity variants.
        variant: One of `VARIANTS`.
        seq_len: Effective sequence length, `s`.

    Returns:
        `(N,)` tensor of FLOPs.
    """
    d = decode(X, variant)
    out = torch.empty(len(d["tokens"]), dtype=torch.double)
    for i, arch in enumerate(d["arch"]):
        out[i] = run_flops(
            arch, d["tokens"][i], d["trainable_layers"][i], seq_len=seq_len
        )
    return out


def _self_check() -> None:
    """Reproduce the data mixture anchor and print the per-config FLOPs table."""
    anchor = run_flops(QWEN3_4B, DMO_TOKENS, QWEN3_4B["num_hidden_layers"]).item()
    rel_err = abs(anchor - DMO_ANCHOR_FLOPS) / DMO_ANCHOR_FLOPS
    print("Data mixture anchor (Qwen3-4B, all layers, 1e7 tokens)")
    print(f"  formula  {anchor:.4g} FLOPs")
    print(f"  expected {DMO_ANCHOR_FLOPS:.4g} FLOPs   ({rel_err:.1%} error)\n")
    assert rel_err < 0.05, f"anchor mismatch: {anchor:.4g} vs {DMO_ANCHOR_FLOPS:.4g}"

    print(f"HPO run FLOPs at the full {FULL_BUDGET_TOKENS:.3g}-token budget")
    print(f"{'lora_layers':>12} {'k':>4} {'Qwen3-4B':>12} {'Qwen3-8B':>12}")
    for opt in LORA_LAYER_OPTIONS:
        k = QWEN3_8B["num_hidden_layers"] if opt == LORA_LAYERS_ALL else opt
        label = "all" if opt == LORA_LAYERS_ALL else f"last_{opt}"
        f4 = run_flops(QWEN3_4B, FULL_BUDGET_TOKENS, k).item()
        f8 = run_flops(QWEN3_8B, FULL_BUDGET_TOKENS, k).item()
        print(f"{label:>12} {k:>4} {f4:>12.3g} {f8:>12.3g}")


if __name__ == "__main__":
    _self_check()
