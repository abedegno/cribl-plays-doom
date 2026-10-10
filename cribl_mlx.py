"""MLX backend for Cribl Decision 1.0 on Apple Silicon.

PyTorch on MPS runs Qwen3.5's Gated DeltaNet layers through transformers'
reference (pure PyTorch) path, about 1 s per Doom decision on an M4 Pro. MLX
has Metal kernels for them. This backend:

  1. merges the LoRA into the pinned base weights in float32
     (W' = W + (lora_alpha / r) * B @ A), casts back to bfloat16, and writes
     the result once to ~/.cache/cribl-doom/merged-<adapter revision>/ in the
     base model's own key layout (so mlx_lm loads it unmodified);
  2. runs exactly the same prompt and readout as cribl_decision.py: one
     forward pass, the logits at the decision slot, softmax over the
     candidates' lm_head rows.

Parity with the PyTorch/PEFT path is checked by tests/test_mlx_parity.py.
Merging changes rounding (bf16(W + BA) instead of bf16(W) + bf16(BA)), so
probabilities match to a tolerance, not bit for bit.
"""
from __future__ import annotations

import json, os, shutil, time
from pathlib import Path
from typing import Any, Mapping

from cribl_decision import (ADAPTER_ID, ADAPTER_REVISION, BASE_ID, BASE_REVISION, FORMATS,
                            Answer, PromptFormat, Question, make_answer)

CACHE = Path(os.environ.get("CRIBL_DOOM_CACHE", Path.home() / ".cache" / "cribl-doom"))


def merged_checkpoint() -> Path:
    out = CACHE / f"merged-{ADAPTER_REVISION[:12]}-{BASE_REVISION[:12]}"
    if (out / "DONE").exists():
        return out
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = Path(snapshot_download(BASE_ID, revision=BASE_REVISION))
    adir = Path(snapshot_download(ADAPTER_ID, revision=ADAPTER_REVISION))
    cfg = json.loads((adir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = {}
    with safe_open(adir / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            # base_model.model.<module>.lora_A.weight -> <module>.weight
            mod, which = k[len("base_model.model."):].rsplit(".lora_", 1)
            lora.setdefault(mod + ".weight", {})[which.split(".")[0]] = f.get_tensor(k).float()
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((base / "model.safetensors.index.json").read_text())
    merged = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(base / shard, "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k in lora:
                    ab = lora[k]
                    t = (t.float() + scale * (ab["B"] @ ab["A"])).to(t.dtype)
                    merged += 1
                tensors[k] = t.contiguous()
        save_file(tensors, out / shard, metadata={"format": "pt"})
    if merged != len(lora):
        raise RuntimeError(f"merged {merged} of {len(lora)} LoRA modules")
    for name in os.listdir(base):
        if not name.endswith(".safetensors") and not (out / name).exists():
            shutil.copy(base / name, out / name)
    (out / "MERGE.json").write_text(json.dumps({
        "base": BASE_ID, "base_revision": BASE_REVISION, "adapter": ADAPTER_ID,
        "adapter_revision": ADAPTER_REVISION, "scale": scale, "modules_merged": merged,
        "method": "float32 W + scale * B @ A, cast to bfloat16"}, indent=1))
    (out / "DONE").write_text("ok\n")
    return out


class CriblDecisionModelMLX:
    """Same interface as cribl_decision.CriblDecisionModel (ask/choose)."""

    def __init__(self, fmt: str | PromptFormat = "plain_letter", temperature: float = 1.0,
                 prefix_cache: bool = True, **_ignored):
        import mlx.core as mx
        from mlx_lm import load
        from transformers import AutoTokenizer
        self.mx = mx
        t0 = time.monotonic()
        path = merged_checkpoint()
        self.model, _ = load(str(path))
        # Same tokenizer object as the torch path so token ids are identical.
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REVISION)
        self.fmt = FORMATS[fmt] if isinstance(fmt, str) else fmt
        self.temperature = temperature
        self.device, self.dtype_name = "mlx-metal", "bfloat16"
        self.adapter_report = json.loads((path / "MERGE.json").read_text())
        self.adapter_report["tensors"] = 2 * self.adapter_report["modules_merged"]
        self.load_seconds = time.monotonic() - t0
        self._row_cache: dict = {}
        self.use_adapter = True
        self.prefix_cache = prefix_cache
        self._prefix: dict = {}       # prefix text -> (ids, cache snapshot)

    def build(self, state: Any, q: Question):
        prompt = self.fmt.prompt(self.tokenizer, state, q)
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        keys = [k for k, _ in q.candidates()]
        key = (self.fmt.name, prompt[-16:], q.type, tuple(keys))
        rows = self._row_cache.get(key)
        if rows is None:
            rows = self._row_cache[key] = self.fmt.candidate_rows(self.tokenizer, prompt, q)
        return ids, rows, keys

    def ask(self, state: Any, questions: Mapping[str, Question]) -> tuple[dict[str, Answer], dict]:
        """Questions are run one sequence at a time (mlx_lm has no padding
        mask); the batched-forward variant lives in the torch backend."""
        mx = self.mx
        t0 = time.monotonic()
        answers, tokens = {}, 0
        fwd = 0.0
        for name, q in questions.items():
            ids, rows, keys = self.build(state, q)
            tokens += len(ids)
            t1 = time.monotonic()
            cache, start = self._cached_prefix(state, q, ids) if self.prefix_cache else (None, 0)
            logits = self.model(mx.array([ids[start:]]), cache=cache)[0, -1].astype(mx.float32)
            logz = mx.logsumexp(logits)
            if all(len(r) == 1 for r in rows):
                cl = logits[mx.array([r[0] for r in rows])]
                mass = mx.exp(mx.logsumexp(cl) - logz)
            else:
                cl = mx.stack([logits[mx.array(r)].mean() for r in rows])
                mass = mx.array(float("nan"))
            mx.eval(cl, mass)
            fwd += time.monotonic() - t1
            answers[name] = make_answer(q, keys, cl.tolist(), float(mass.item()), self.temperature)
        t2 = time.monotonic()
        return answers, {"prep_ms": (t2 - t0 - fwd) * 1000, "forward_ms": fwd * 1000,
                         "total_ms": (t2 - t0) * 1000, "tokens": tokens, "batch": len(questions)}

    def _cached_prefix(self, state: Any, q: Question, ids: list[int]):
        """Reuse the model state after the static (state-independent) prefix.
        Only formats that put the state last have one. Returns (cache, n) where
        the first n prompt tokens are already in the cache."""
        from mlx_lm.models.cache import KVCache
        mx = self.mx
        prefix = self.fmt.static_prefix(state, q)
        if not prefix:
            return None, 0
        hit = self._prefix.get(prefix)
        if hit is None:
            pids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            cache = self.model.make_cache()
            mx.eval(self.model(mx.array([pids]), cache=cache))
            snap = []
            for c in cache:
                if isinstance(c, KVCache):
                    # slice to exactly `offset` so later updates reallocate
                    # instead of writing into the shared buffer
                    k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
                    mx.eval(k, v)
                    snap.append(("kv", k, v, c.offset))
                else:
                    # copy the list: layers assign into it (cache[0] = ...)
                    mx.eval(*[a for a in c.cache if a is not None])
                    snap.append(("arr", list(c.cache)))
            hit = self._prefix[prefix] = (pids, snap)
        pids, snap = hit
        if ids[:len(pids)] != pids:          # tokenisation differs across the cut
            return None, 0
        cache = self.model.make_cache()
        for c, s in zip(cache, snap):
            if s[0] == "kv":
                c.keys, c.values, c.offset = s[1], s[2], s[3]
            else:
                c.cache = list(s[1])
        return cache, len(pids)

    def choose(self, state: Any, q: Question):
        a, t = self.ask(state, {"q": q})
        return a["q"], t
