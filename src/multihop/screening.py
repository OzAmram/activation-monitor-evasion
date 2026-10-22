"""Model-based screen for nonsensical source facts (e.g. Wikidata errors like "the founder of Ralph Lauren
Corporation is Latvia"). Judges only whether the object is the right *kind* of thing for the relation, not
whether the fact is true. Uses a fixed judge model from a different family than the evaluated models, and is
run once per dataset version, so the resulting filter is model-independent for the experiments.
"""
import torch
from tqdm import tqdm

from multihop.model import tokenize

SCREEN_PROMPT = (
    "The following statement comes from an automatically built knowledge base that contains some errors:\n\n"
    "\"{np} is {obj}.\"\n\n"
    "Does this statement make sense, i.e. is \"{obj}\" the right kind of thing to be {np}? For example, the "
    "founder of a company should be a person or organization, not a country, and a capital should be a city. "
    "You do not need to know whether the statement is actually true, only whether it makes sense. "
    "Answer with only Yes or No."
)


def screen_prompts(tokenizer, facts: list[tuple[str, str]]) -> list[str]:
    msgs = [[{"role": "user", "content": SCREEN_PROMPT.format(np=np_[0].upper() + np_[1:], obj=obj)}] for np_, obj in facts]
    # Qwen3: disable thinking so the answer is the next token.
    return [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]


@torch.no_grad()
def p_yes(model, tokenizer, prompts: list[str], batch_size: int = 64) -> list[float]:
    """P(Yes) / (P(Yes) + P(No)) for the next token after each prompt."""
    yes, no = (tokenizer.encode(w, add_special_tokens=False) for w in ("Yes", "No"))
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Yes/No are not single tokens for this tokenizer")
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # length-sorted batches, less padding
    out = [0.0] * len(prompts)
    for s in tqdm(range(0, len(order), batch_size), desc="screen"):
        idx = order[s:s + batch_size]
        batch = tokenize(tokenizer, [prompts[i] for i in idx], model.get_input_embeddings().weight.device)
        logits = model(**batch, use_cache=False).logits[:, -1, :].float()
        pair = torch.softmax(logits[:, [yes[0], no[0]]], dim=-1)[:, 0].cpu().tolist()
        for i, p in zip(idx, pair):
            out[i] = p
    return out
