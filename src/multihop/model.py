"""Model loading, prompt formatting, residual-stream recording, and interventions.

Backend: plain HuggingFace transformers + forward hooks on decoder layers.
Everything the rest of the project needs goes through the small set of public
functions below (load_model, format_prompts, record, greedy_generate,
ablate_direction, add_direction, layer helpers), so the backend can later be
swapped (e.g. for nnsight or vLLM) without touching callers.

Conventions:
- "layer i" means the OUTPUT of decoder layer i (0-based), i.e. the residual
  stream after that layer. It equals HF `output_hidden_states[i + 1]`.
- Layers in configs are given as fractions of depth (see `layer_at_fraction`,
  `layers_in_band`), so configs transfer across models with different depths.
- Prompts are left-padded, so the last K positions of every row are the last
  K real prompt tokens.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ----------------------------------------------------------------------------- config


@dataclass
class ModelConfig:
    model_id: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    k_positions: int = 8  # number of final prompt tokens to record / intervene on
    store_layers: list[float] | None = None  # fractions of depth; None = all layers
    middle_band: list[float] = field(default_factory=lambda: [0.25, 0.75])
    seed: int = 0
    batch_size: int = 16


def load_config(path: str) -> ModelConfig:
    with open(path) as f:
        return ModelConfig(**yaml.safe_load(f))


# ----------------------------------------------------------------------------- loading


def load_model(cfg: ModelConfig):
    """Load model + tokenizer. Tokenizer is set to left padding with a pad token."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # Padded positions are masked out, so reusing EOS as pad is safe.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=DTYPES[cfg.dtype], device_map=cfg.device_map
    )
    model.eval()
    return model, tokenizer


def get_layers(model) -> torch.nn.ModuleList:
    """Decoder layers of a standard HF causal LM (`model.model.layers`). Fails loudly otherwise."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        raise ValueError(
            f"Cannot find decoder layers at model.model.layers for {type(model).__name__}; "
            "add support for this architecture in get_layers()."
        )
    return layers


def n_layers(model) -> int:
    return len(get_layers(model))


def layer_at_fraction(frac: float, num_layers: int) -> int:
    """Layer index at a fraction of depth: round(frac * num_layers), clipped to the last layer."""
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {frac}")
    return min(int(frac * num_layers + 0.5), num_layers - 1)


def layers_in_band(band: list[float], num_layers: int) -> list[int]:
    """Layer indices in a half-open band [lo, hi) of depth fractions.

    Uses round(lo * n) .. round(hi * n) - 1, so [0, 1] gives all layers and
    [0.25, 0.75] on 32 layers gives layers 8..23 (16 layers).
    """
    lo, hi = band
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"band must satisfy 0 <= lo < hi <= 1, got {band}")
    start, stop = int(lo * num_layers + 0.5), int(hi * num_layers + 0.5)
    return list(range(start, max(stop, start + 1)))


def resolve_store_layers(cfg: ModelConfig, num_layers: int) -> list[int]:
    if cfg.store_layers is None:
        return list(range(num_layers))
    return sorted({layer_at_fraction(f, num_layers) for f in cfg.store_layers})


# ----------------------------------------------------------------------------- prompts


def format_prompts(tokenizer, user_messages: list[str]) -> list[str]:
    """Wrap each user message in the chat template, ending with the assistant generation prompt.

    Note: no system message is passed, so the template's default system prompt (if any)
    is used. Olmo 3 Instruct inserts one ("You are a helpful function-calling AI assistant...").
    """
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True
        )
        for m in user_messages
    ]


def tokenize(tokenizer, prompts: list[str], device) -> dict:
    """Left-padded batch. The chat template already contains any BOS, so no special tokens are added."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    mask = enc["attention_mask"]
    # Position ids that ignore left padding (so a padded row matches the same prompt run alone).
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": mask.to(device),
        "position_ids": position_ids.to(device),
    }


def _input_device(model):
    return model.get_input_embeddings().weight.device


def _hidden(output):
    """Decoder layers return either a tensor or a tuple whose first element is the hidden state."""
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, new_hidden):
    return (new_hidden,) + tuple(output[1:]) if isinstance(output, tuple) else new_hidden


# ----------------------------------------------------------------------------- recording


@dataclass
class Recording:
    acts: torch.Tensor  # [batch, n_layers_recorded, K, hidden], CPU, model dtype
    layers: list[int]  # layer indices along dim 1
    read_tokens: list[list[str]]  # token strings at the K read positions, per item
    next_token_ids: torch.Tensor  # [batch] greedy next token after the prompt
    next_tokens: list[str]  # decoded greedy next token
    last_logits: torch.Tensor  # [batch, vocab] float32 CPU, logits at the last prompt position
    topk_ids: torch.Tensor  # [batch, topk]
    topk_logprobs: torch.Tensor  # [batch, topk] float32


@torch.no_grad()
def record(model, tokenizer, prompts: list[str], k: int, layers: list[int] | None = None, topk: int = 10) -> Recording:
    """Run one forward pass on a batch of formatted prompts.

    Records the residual stream (output of each layer in `layers`, default all) at the last
    `k` prompt tokens, plus next-token logits at the last position. Interventions
    (ablate_direction / add_direction) active around this call are applied.
    """
    all_layers = get_layers(model)
    layers = list(range(len(all_layers))) if layers is None else list(layers)
    batch = tokenize(tokenizer, prompts, _input_device(model))
    if k > batch["input_ids"].shape[1] or not bool(batch["attention_mask"][:, -k:].all()):
        raise ValueError(f"Some prompt is shorter than k={k} tokens.")

    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook(module, inputs, output):
            captured[idx] = _hidden(output)[:, -k:, :].detach().to("cpu")

        return hook

    handles = [all_layers[i].register_forward_hook(make_hook(i)) for i in layers]
    try:
        out = model(**batch, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    acts = torch.stack([captured[i] for i in layers], dim=1)  # [B, L, K, H]
    last_logits = out.logits[:, -1, :].float().cpu()
    logprobs = torch.log_softmax(last_logits, dim=-1)
    topk_lp, topk_ids = logprobs.topk(topk, dim=-1)
    next_ids = last_logits.argmax(-1)
    read_ids = batch["input_ids"][:, -k:].cpu()
    return Recording(
        acts=acts,
        layers=layers,
        read_tokens=[tokenizer.convert_ids_to_tokens(row) for row in read_ids.tolist()],
        next_token_ids=next_ids,
        next_tokens=[tokenizer.decode([i]) for i in next_ids.tolist()],
        last_logits=last_logits,
        topk_ids=topk_ids,
        topk_logprobs=topk_lp,
    )


@torch.no_grad()
def greedy_generate(model, tokenizer, prompts: list[str], max_new_tokens: int = 16) -> list[str]:
    """Greedy decoding of a short answer for each formatted prompt. Returns only the new text.

    Overrides the model's generation config (Olmo 3 defaults to sampling) with do_sample=False.
    """
    batch = tokenize(tokenizer, prompts, _input_device(model))
    del batch["position_ids"]  # generate() derives positions from the attention mask
    out = model.generate(
        **batch,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[:, batch["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


# ----------------------------------------------------------------------------- interventions


@contextmanager
def ablate_direction(model, directions, layers: list[int], k: int):
    """Project a unit direction out of the residual stream at the last `k` positions of `layers`.

    `directions`: a [hidden] tensor (same direction at every layer) or a dict {layer: [hidden]}.
    Directions are normalised here. Applied to the output of each listed layer, at the last
    `k` positions of every forward call. Intended for single forward passes over prompts
    (e.g. `record`); inside `generate` with a KV cache, later calls have 1 position, so the
    ablation would also hit each generated token.
    """
    all_layers = get_layers(model)

    def unit(d):
        d = d.detach().float()
        return d / d.norm()

    per_layer = {l: unit(directions[l] if isinstance(directions, dict) else directions) for l in layers}

    def make_hook(idx):
        def hook(module, inputs, output):
            h = _hidden(output)
            d = per_layer[idx].to(h.device)
            tail = h[:, -k:, :].float()
            tail = tail - (tail @ d).unsqueeze(-1) * d
            new_h = h.clone()
            new_h[:, -k:, :] = tail.to(h.dtype)
            return _replace_hidden(output, new_h)

        return hook

    handles = []
    try:
        for i in layers:
            handles.append(all_layers[i].register_forward_hook(make_hook(i)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def add_direction(model, direction: torch.Tensor, layer: int, scale: float):
    """Steering: add `scale * direction` to the output of `layer` at all positions.

    The direction is used as given (not normalised), so `scale` is in units of its norm.
    Also applies to padded positions, which is harmless because they are masked out.
    """
    target = get_layers(model)[layer]
    vec = (scale * direction.detach().float())

    def hook(module, inputs, output):
        h = _hidden(output)
        return _replace_hidden(output, h + vec.to(h.device, h.dtype))

    handle = target.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
