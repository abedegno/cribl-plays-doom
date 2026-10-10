"""Cribl Decision 1.0, run locally, with a structured candidate readout.

The model is a LoRA adapter (cribl-ai/cribl-decision-1.0) on
Qwen/Qwen3.5-4B-Base. The model card describes the readout as:

    hidden state at the decision slot
    -> native Qwen lm_head candidate logits
    -> normalization over the supplied candidates
    -> typed choice / noul / score output

and conceptually `candidate_logits = candidate_weights @ decision_hidden_state`.

Cribl has not published the "accompanying candidate-scoring implementation"
(no code in the HF repo, none on GitHub as of 2026-10-10). So the parts that
are NOT specified by the card -- the prompt text, where the decision slot is,
and which lm_head rows are a candidate's `candidate_weights` -- are explicit,
swappable `PromptFormat`s here. `probe_formats.py` measures which format the
adapter actually responds to (accuracy and adapter-vs-base uplift on probes
with known answers). Nothing in this module generates or parses text: one
forward pass, read one position, softmax over the candidates' rows.

What IS taken from the card and verified in tests:
  * base model + pinned revision, adapter loaded with PEFT;
  * every adapter tensor is actually applied (PEFT only warns on a key
    mismatch, which would silently run the bare base model);
  * logits come from the native lm_head rows (tied embeddings in this model);
  * probabilities are a softmax over the supplied candidates only.
"""
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

BASE_ID = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"   # pinned by the model card
ADAPTER_ID = "cribl-ai/cribl-decision-1.0"
ADAPTER_REVISION = "75cb0dd9d551e774664c01ba8f7b25c0087b4abf"  # HF main on 2026-10-10

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------------------
# Typed questions and answers (the Jev / systemone request shape)
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """One typed question. `criteria` is a name->description mapping for
    choice, an ordered list of level descriptions for score, and optional
    {"true": ..., "false": ...} for noul -- the same shapes as /v1/systemone."""
    type: str
    instructions: str
    criteria: Any = None

    def candidates(self) -> list[tuple[str, str | None]]:
        if self.type == "choice":
            if not isinstance(self.criteria, Mapping) or not 1 <= len(self.criteria) <= 255:
                raise ValueError("choice needs 1..255 named candidates")
            return [(str(k), None if v is None else str(v)) for k, v in self.criteria.items()]
        if self.type == "score":
            if not isinstance(self.criteria, Sequence) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score needs 2..10 ordered levels")
            return [(str(i), str(v)) for i, v in enumerate(self.criteria)]
        if self.type == "noul":
            c = self.criteria or {}
            return [("true", c.get("true")), ("false", c.get("false"))]
        raise ValueError(f"unknown question type {self.type!r}")


@dataclass
class Answer:
    type: str
    probabilities: dict[str, float]
    choice: str | None = None        # choice: argmax candidate
    noul: float | None = None        # noul: P(true)
    score: float | None = None       # score: expected level
    confidence: float = 0.0          # official system-one formula (see choice_confidence)
    p_selected: float = 0.0          # probability of the argmax candidate
    candidate_mass: float = 0.0      # share of the full-vocab softmax on candidate rows
    logits: list[float] = field(default_factory=list)


def softmax(xs: Sequence[float], temperature: float = 1.0) -> list[float]:
    m = max(xs)
    es = [math.exp((x - m) / temperature) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def choice_confidence(probs: Sequence[float]) -> float:
    """(max - 1/n) / (1 - 1/n): 0 for a uniform distribution, 1 for a point
    mass. Same formula Open-Jev records as the official system-one adapter's."""
    n = len(probs)
    if n == 1:
        return 1.0
    u = 1.0 / n
    return (max(probs) - u) / (1.0 - u)


def score_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    if n == 1:
        return 1.0
    mode = max(range(n), key=probs.__getitem__)
    dist = sum(p * abs(i - mode) for i, p in enumerate(probs))
    centre = (n - 1) / 2
    uniform_dev = sum(abs(i - centre) for i in range(n)) / n
    return max(0.0, 1.0 - dist / uniform_dev)


def make_answer(q: Question, keys: list[str], logits: list[float], mass: float,
                temperature: float = 1.0) -> Answer:
    p = softmax(logits, temperature)
    probs = dict(zip(keys, p))
    best = max(range(len(p)), key=p.__getitem__)
    a = Answer(type=q.type, probabilities=probs, p_selected=p[best],
               candidate_mass=mass, logits=list(logits))
    if q.type == "choice":
        a.choice = keys[best]
        a.confidence = choice_confidence(p)
    elif q.type == "noul":
        a.noul = probs["true"]
        a.choice = keys[best]
        a.confidence = choice_confidence(p)
    else:
        a.score = sum(i * x for i, x in enumerate(p))
        a.choice = keys[best]
        a.confidence = score_confidence(p)
    return a


# ---------------------------------------------------------------------------
# Prompt formats: where the decision slot is and which lm_head rows score
# ---------------------------------------------------------------------------

def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(", ", ": "))


@dataclass(frozen=True)
class PromptFormat:
    """How a (state, question, candidates) triple becomes token ids ending at
    the decision slot, and which vocabulary rows each candidate is scored by.

    template:  "chat"  -> Qwen chat template, thinking disabled, slot is the
                          first assistant position;
               "plain" -> raw text ending in "Answer:".
    labels:    "letter" -> candidates listed as "A. name: description", scored
                           by the row of the token that spells the letter;
               "name"   -> listed by name, scored by the first token of the
                           name (rejected if two names share a first token);
               "name_mean" -> listed by name, scored by the mean of the rows
                              of all the name's tokens.
    """
    name: str
    template: str = "chat"
    labels: str = "letter"
    state_first: bool = True

    def option_label(self, i: int, key: str) -> str:
        return LETTERS[i] if self.labels == "letter" else key

    def body(self, state: Any, q: Question) -> str:
        cands = q.candidates()
        if self.labels == "letter" and len(cands) > len(LETTERS):
            raise ValueError("letter labels support at most 26 candidates")
        if q.type == "noul":
            lines = ["A. Yes" + (f": {cands[0][1]}" if cands[0][1] else ""),
                     "B. No" + (f": {cands[1][1]}" if cands[1][1] else "")] \
                if self.labels == "letter" else \
                ["Yes" + (f": {cands[0][1]}" if cands[0][1] else ""),
                 "No" + (f": {cands[1][1]}" if cands[1][1] else "")]
        else:
            lines = []
            for i, (k, d) in enumerate(cands):
                lab = self.option_label(i, k)
                if self.labels == "letter":
                    lines.append(f"{lab}. {k}: {d}" if d and q.type == "choice" else
                                 f"{lab}. {d if d else k}")
                else:
                    lines.append(f"{k}: {d}" if d else k)
        what = "letter" if self.labels == "letter" else "candidate"
        parts_state = f"State:\n{render_state(state)}"
        parts_q = (f"Question ({q.type}): {q.instructions}\n\n"
                   f"Candidates:\n" + "\n".join(lines) +
                   f"\n\nAnswer with the {what} of the single best candidate.")
        return (parts_state + "\n\n" + parts_q) if self.state_first else \
               (parts_q + "\n\n" + parts_state)

    def static_prefix(self, state: Any, q: Question) -> str | None:
        """The part of the prompt that does not depend on the state, when the
        state comes last (so its KV / recurrent state can be cached)."""
        if self.state_first or self.template != "plain":
            return None
        body = self.body(state, q)
        cut = body.index("State:\n")
        return body[:cut]

    def prompt(self, tokenizer, state: Any, q: Question) -> str:
        body = self.body(state, q)
        if self.template == "chat":
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": body}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        return body + "\nAnswer:"

    def candidate_surface(self, i: int, key: str, q: Question) -> str:
        """The text the model would emit at the slot for candidate i."""
        if q.type == "noul":
            word = ("A" if i == 0 else "B") if self.labels == "letter" else ("Yes" if i == 0 else "No")
        else:
            word = LETTERS[i] if self.labels == "letter" else key
        # In the plain template the slot follows "Answer:" so the candidate is
        # spelt with a leading space; after the chat template's "\n\n" it is not.
        return (" " + word) if self.template == "plain" else word

    def candidate_rows(self, tokenizer, prompt: str, q: Question) -> list[list[int]]:
        """Token ids whose lm_head rows score each candidate. Ids are found by
        tokenising prompt+surface and diffing against the prompt, so merges
        across the slot boundary are respected."""
        base = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        rows = []
        for i, (k, _) in enumerate(q.candidates()):
            full = tokenizer(prompt + self.candidate_surface(i, k, q),
                             add_special_tokens=False)["input_ids"]
            if full[:len(base)] != base:
                raise ValueError(f"candidate {k!r} merges with the prompt tail")
            ids = full[len(base):]
            if not ids:
                raise ValueError(f"candidate {k!r} has no tokens")
            rows.append(ids if self.labels == "name_mean" else ids[:1])
        if self.labels != "name_mean":
            firsts = [r[0] for r in rows]
            if len(set(firsts)) != len(firsts):
                raise ValueError(f"format {self.name}: candidates share a scoring "
                                 f"token ({firsts}); use letter labels")
        return rows


FORMATS: dict[str, PromptFormat] = {f.name: f for f in [
    PromptFormat("chat_letter", "chat", "letter"),
    PromptFormat("plain_letter", "plain", "letter"),
    PromptFormat("chat_name", "chat", "name"),
    PromptFormat("plain_name", "plain", "name"),
    PromptFormat("chat_name_mean", "chat", "name_mean"),
    PromptFormat("chat_letter_qfirst", "chat", "letter", state_first=False),
    PromptFormat("plain_letter_qfirst", "plain", "letter", state_first=False),
]}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def pick_device(pref: str = "auto") -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CriblDecisionModel:
    """Loads base + adapter once; answers typed questions by candidate scoring."""

    def __init__(self, device: str = "auto", dtype: str = "bfloat16",
                 fmt: str | PromptFormat = "plain_letter", temperature: float = 1.0,
                 use_adapter: bool = True):
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from peft import PeftModel

        self.torch = torch
        self.device = pick_device(device)
        self.dtype = getattr(torch, dtype)
        self.fmt = FORMATS[fmt] if isinstance(fmt, str) else fmt
        self.temperature = temperature
        t0 = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REVISION)
        # NOT AutoModelForCausalLM: in transformers 5.x that resolves to the
        # text-only Qwen3_5ForCausalLM whose modules are `model.layers.*`, while
        # the adapter was saved against `model.language_model.layers.*`. PEFT
        # then warns and runs the bare base model. _verify_adapter() catches it.
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            BASE_ID, revision=BASE_REVISION, dtype=self.dtype)
        base.model.visual = None   # vision tower is never used (text-only state)
        self.model = PeftModel.from_pretrained(base, ADAPTER_ID, revision=ADAPTER_REVISION)
        self.model.to(self.device).eval()
        self.adapter_report = self._verify_adapter()
        self.use_adapter = use_adapter
        self.load_seconds = time.monotonic() - t0
        self.lm_head = self.model.get_output_embeddings().weight   # [vocab, hidden]
        self._row_cache: dict[tuple, list[list[int]]] = {}

    # -- integrity --------------------------------------------------------
    def _verify_adapter(self) -> dict:
        """Prove every tensor in adapter_model.safetensors is in the live model
        with the same values. PEFT only *warns* on unmatched keys."""
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        path = hf_hub_download(ADAPTER_ID, "adapter_model.safetensors", revision=ADAPTER_REVISION)
        live = {k: v for k, v in self.model.state_dict().items() if "lora_" in k}
        # live keys carry the adapter name: ...lora_A.default.weight
        live = {k.replace(".default.weight", ".weight"): v for k, v in live.items()}
        n, worst = 0, 0.0
        with safe_open(path, "pt") as f:
            keys = list(f.keys())
            for k in keys:
                if k not in live:
                    raise RuntimeError(f"adapter tensor not applied to model: {k}")
                ref = f.get_tensor(k).float()
                worst = max(worst, (live[k].detach().float().cpu() - ref).abs().max().item())
                n += 1
        if worst > 1e-2:
            raise RuntimeError(f"adapter tensors differ from file (max abs {worst})")
        return {"tensors": n, "max_abs_diff": worst}

    # -- readout ----------------------------------------------------------
    def _rows(self, prompt_tail_key: tuple, prompt: str, q: Question) -> list[list[int]]:
        rows = self._row_cache.get(prompt_tail_key)
        if rows is None:
            rows = self.fmt.candidate_rows(self.tokenizer, prompt, q)
            self._row_cache[prompt_tail_key] = rows
        return rows

    def build(self, state: Any, q: Question) -> tuple[list[int], list[list[int]], list[str]]:
        prompt = self.fmt.prompt(self.tokenizer, state, q)
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        keys = [k for k, _ in q.candidates()]
        # Rows depend only on the prompt's last few characters and the labels.
        rows = self._rows((self.fmt.name, prompt[-16:], q.type, tuple(keys)), prompt, q)
        return ids, rows, keys

    def ask(self, state: Any, questions: Mapping[str, Question]) -> tuple[dict[str, Answer], dict]:
        """Answer every question in ONE batched forward pass (one sequence per
        question, left-padded so the decision slot is the last column)."""
        torch = self.torch
        t0 = time.monotonic()
        built = {name: self.build(state, q) for name, q in questions.items()}
        names = list(built)
        seqs = [built[n][0] for n in names]
        L = max(len(s) for s in seqs)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=self.device)
        attn = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=self.device)
        t1 = time.monotonic()
        with torch.inference_mode():
            ctx = self.model.disable_adapter() if not self.use_adapter else _null()
            with ctx:
                out = self.model(input_ids=input_ids, attention_mask=attn,
                                 logits_to_keep=1, use_cache=False)
            logits = out.logits[:, -1, :].float()                      # [B, vocab]
            logz = torch.logsumexp(logits, dim=-1)
            answers = {}
            for b, n in enumerate(names):
                _, rows, keys = built[n]
                if all(len(r) == 1 for r in rows):
                    cl = logits[b, [r[0] for r in rows]]
                    mass = torch.exp(torch.logsumexp(cl, 0) - logz[b]).item()
                else:
                    # name_mean: mean of the candidate's lm_head rows, applied
                    # to the slot hidden state. lm_head has no bias, so this is
                    # the mean of the candidate tokens' logits.
                    cl = torch.stack([logits[b, r].mean() for r in rows])
                    mass = float("nan")
                answers[n] = make_answer(questions[n], keys, cl.tolist(), mass, self.temperature)
            if self.device == "mps":
                torch.mps.synchronize()
        t2 = time.monotonic()
        timing = {"prep_ms": (t1 - t0) * 1000, "forward_ms": (t2 - t1) * 1000,
                  "total_ms": (t2 - t0) * 1000, "tokens": int(attn.sum().item()),
                  "batch": len(names)}
        return answers, timing

    def choose(self, state: Any, q: Question) -> tuple[Answer, dict]:
        a, t = self.ask(state, {"q": q})
        return a["q"], t


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False
