"""Concept directions (SPEC 4; Lindsey 2026 arXiv:2601.01828, Kowalski et al. 2026 arXiv:2608.21664 App. A).

For entity x and layer l: residual stream at the last prompt token of the chat-formatted "Tell me about {x}",
minus the mean of the same quantity over the 99-word baseline pool. Returned un-normalised; normalise for
projections/ablations.
"""
from pathlib import Path

import torch

from multihop.eligibility import names_match
from multihop.model import add_direction, format_prompts, get_layers, greedy_generate, record

DIRECTION_PROMPT = "Tell me about {x}"


def load_baseline_words(path: str | Path) -> list[str]:
    words = [w.strip() for w in Path(path).read_text().splitlines() if w.strip() and not w.startswith("#")]
    if len(words) != len(set(words)):
        raise ValueError("duplicate baseline words")
    return words


@torch.no_grad()
def last_token_acts(model, tok, names: list[str], layers: list[int], batch_size: int) -> torch.Tensor:
    """[n, n_layers, hidden] float32 residual at the last prompt token of "Tell me about {x}"."""
    prompts = format_prompts(tok, [DIRECTION_PROMPT.format(x=x) for x in names])
    return record(model, tok, prompts, k=1, layers=layers, batch_size=batch_size).acts[:, :, 0, :].float()


def concept_vectors(model, tok, names: list[str], baseline_words: list[str], layers: list[int], batch_size: int) -> torch.Tensor:
    """[n, n_layers, hidden] float32 concept vectors (entity activation minus baseline mean)."""
    base = last_token_acts(model, tok, baseline_words, layers, batch_size).mean(0)
    return last_token_acts(model, tok, names, layers, batch_size) - base


def mean_resid_norm(model, tok, layer: int, batch_size: int, words: list[str]) -> float:
    """Typical residual norm at `layer` (last prompt token over baseline prompts), to set steering scales."""
    return last_token_acts(model, tok, words, [layer], batch_size)[:, 0].norm(dim=-1).mean().item()


def steering_check(model, tok, name: str, aliases: list[str], vec: torch.Tensor, layer: int, scales: list[float],
                   prompt: str = "Tell me what you are thinking about right now, in one sentence.",
                   max_new_tokens: int = 40) -> list[dict]:
    """Add scale * unit(vec) at `layer` (all positions) and check whether the generation names the entity.
    `scales` are absolute norms of the added vector (use multiples of mean_resid_norm)."""
    unit = vec / vec.norm()
    p = format_prompts(tok, [prompt])
    res = []
    for sc in scales:
        if sc == 0:
            text = greedy_generate(model, tok, p, max_new_tokens)[0]
        else:
            with add_direction(model, unit, layer, sc):
                text = greedy_generate(model, tok, p, max_new_tokens)[0]
        res.append({"entity": name, "layer": layer, "scale": sc, "mentions": names_match(text, list(aliases)), "text": text})
    return res
