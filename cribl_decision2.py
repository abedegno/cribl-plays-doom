"""Cribl Decision 2, run locally: Cribl's own reader, plus an MLX port of it.

Decision 2 is not a LoRA like 1.0. Each repository is a full Qwen3.5 text
backbone plus `decision_head.safetensors` (an option scorer) and Cribl's
reference inference code, `decision.py`. So there is nothing to guess this
time: the prompt, the readout position and the scoring are all defined by
that file.

Two backends:

  torch  Cribl's `decision.py`, imported unmodified from the pinned Hugging
         Face snapshot. This is the reference. On Apple Silicon it runs the
         linear-attention layers through transformers' pure PyTorch fallback,
         about 1.2 s per Doom decision for the 4B on an M4 Pro.

  mlx    The same network on mlx_lm's Qwen3.5 layers, which have Metal kernels.
         Three changes from the stock causal model, exactly as decision.py
         makes them: the full-attention layers get no causal mask; the
         linear-attention layers in `reverse_scan_layers` also read the prompt
         right to left, added through tanh(gate); and the option scorer
         replaces the LM head. The prompt text and token ids come from
         decision.py's own `prompt()` method, so they cannot drift.
         tests/test_decision2.py checks it against the torch reference.

Both return the same `Decision2Answer`.
"""
from __future__ import annotations

import importlib.util, json, math, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Pinned: HF main on 2026-11-18. The model card's default repo id
# ("cribl-ai/cribl-decision-2") is not where the weights are published; the
# sizes live in separate repositories.
MODELS = {
    "4b": ("cribl-ai/cribl-decision-2-4B", "317e53b77f62011f442f56cb7c5032484fadd850"),
    "mini": ("cribl-ai/cribl-decision-2-0.8B", "fede0fcd51b2fcbace7ed1f1729bb404eddfeb2a"),
}
LABEL = {"4b": "Cribl Decision 2 (4B)", "mini": "Cribl Decision 2 Mini (0.8B)"}


def snapshot(size: str) -> Path:
    from huggingface_hub import snapshot_download
    repo, rev = MODELS[size]
    return Path(snapshot_download(repo, revision=rev))


def reference_module(path: Path):
    """Import the repository's decision.py under a private name. It must be in
    sys.modules before it runs, or its dataclasses cannot resolve their module."""
    name = f"_cribl_decision2_ref_{path.name[:12]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path / "decision.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def prompt_shell(path: Path):
    """decision.py's own DecisionModel, with its tokenizer, config and prompt
    code but no weights: the same __init__ steps that concern the prompt, so
    prompt() and the marker ids are exactly the reference's."""
    import torch.nn as tnn
    from transformers import AutoTokenizer
    ref = reference_module(path)
    config = json.loads((path / "decision_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "right"
    shell = ref.DecisionModel.__new__(ref.DecisionModel)
    tnn.Module.__init__(shell)
    shell.tokenizer, shell.config = tok, config
    added = [t.content for t in getattr(tok, "added_tokens_decoder", {}).values()]
    shell._escapes = sorted({*tok.all_special_tokens, *added, config["readout_token"],
                             *config["reserved_tokens"]}, key=len, reverse=True)
    shell.readout_id = shell._single(config["readout_token"])
    shell.choice_end_id = shell._single(config["choice_end_token"])
    return shell


@dataclass
class Decision2Answer:
    choice: str
    probabilities: dict[str, float]
    confidence: float      # (max p - 1/n) / (1 - 1/n), the same formula as the 1.0 code
    p_selected: float


def make_answer(probs: Mapping[str, float]) -> Decision2Answer:
    n = len(probs)
    best = max(probs, key=probs.get)
    p = probs[best]
    conf = (p - 1 / n) / (1 - 1 / n) if n > 1 else 1.0
    return Decision2Answer(best, dict(probs), conf, p)


class _Base:
    size: str
    device: str
    dtype_name = "bfloat16"
    load_seconds: float
    revision: str

    def decide(self, state: Any, question: str, options: Mapping[str, str | None]
               ) -> tuple[Decision2Answer, dict]:
        raise NotImplementedError

    @property
    def adapter_report(self) -> dict:      # what doom_agent prints after loading
        return {"repo": MODELS[self.size][0], "revision": self.revision}


class Decision2Torch(_Base):
    """Cribl's reference implementation, unmodified."""

    def __init__(self, size: str = "4b", device: str | None = None):
        t0 = time.monotonic()
        self.size, self.revision = size, MODELS[size][1]
        path = snapshot(size)
        self.ref = reference_module(path)
        self.model = self.ref.DecisionModel.from_pretrained(path, device=None if device in (None, "auto") else device)
        self.device = f"torch-{next(self.model.parameters()).device.type}"
        self.load_seconds = time.monotonic() - t0

    def decide(self, state, question, options):
        t0 = time.monotonic()
        d = self.model.decide(state=state, question=question, options=dict(options))
        return make_answer(d.probabilities), {"total_ms": (time.monotonic() - t0) * 1000}


class Decision2MLX(_Base):
    """The same network on MLX. See the module docstring for what differs from
    a stock causal Qwen3.5 and why each change matches decision.py."""

    def __init__(self, size: str = "4b"):
        import mlx.core as mx
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        t0 = time.monotonic()
        self.mx, self.size, self.revision = mx, size, MODELS[size][1]
        path = snapshot(size)
        self.config = json.loads((path / "decision_config.json").read_text())
        hf = json.loads((path / "config.json").read_text())
        if hf["architectures"] != ["Qwen3_5TextModel"]:
            raise ValueError(f"unexpected backbone {hf['architectures']}")

        # Backbone. mlx_lm's sanitize() converts HF conv1d layout and the
        # (1 + w) RMSNorm convention; it expects the "model." prefix.
        self.net = TextModel(TextModelArgs.from_dict(hf))
        weights = {}
        for f in sorted(path.glob("model*.safetensors")):
            weights.update({"model." + k: v for k, v in mx.load(str(f)).items()})
        weights = self.net.sanitize(weights)
        self.net.load_weights(list(weights.items()), strict=True)
        self.net.eval()

        head = mx.load(str(path / "decision_head.safetensors"))
        self.query, self.key = head["query"], head["key"]          # (rank, hidden)
        self.rank = self.query.shape[0]
        self.reverse = {i: mx.tanh(head[f"reverse_gate.{i}"].reshape(()).astype(mx.float32))
                        for i in self.config["reverse_scan_layers"]}
        for i in self.reverse:
            if not self.net.model.layers[i].is_linear:
                raise ValueError(f"reverse-scan layer {i} is not a linear-attention layer")
        mx.eval(self.net.parameters(), self.query, self.key, *self.reverse.values())

        self.ref = reference_module(path)
        self.shell = prompt_shell(path)
        self.tokenizer = self.shell.tokenizer
        self.max_tokens = int(self.config["max_sequence_tokens"])
        self.device = "mlx-metal"
        self.load_seconds = time.monotonic() - t0

    def encode(self, state, question, options) -> tuple[list[int], int, list[int], list[str]]:
        """Token ids, readout position, option-marker positions, labels: the
        same checks decision_batch() makes before it runs the model."""
        q = self.ref.Question(state, question, dict(options))
        text, labels = self.shell.prompt(q)
        ids = self.tokenizer(text)["input_ids"]
        if len(ids) > self.max_tokens:
            raise ValueError(f"prompt has {len(ids)} tokens; the limit is {self.max_tokens}")
        readouts = [i for i, t in enumerate(ids) if t == self.shell.readout_id]
        ends = [i for i, t in enumerate(ids) if t == self.shell.choice_end_id]
        if len(readouts) != 1 or len(ends) != len(labels):
            raise ValueError("prompt markers do not match the options")
        return ids, readouts[0], ends, labels

    def hidden(self, ids: list[int]):
        """Final hidden states for one unpadded sequence."""
        mx = self.mx
        m = self.net.model
        h = m.embed_tokens(mx.array([ids]))
        for i, layer in enumerate(m.layers):
            x = layer.input_layernorm(h)
            if layer.is_linear:
                r = layer.linear_attn(x, None, None)
                if i in self.reverse:     # read right to left with the same weights
                    back = layer.linear_attn(x[:, ::-1, :], None, None)[:, ::-1, :]
                    r = r + self.reverse[i].astype(r.dtype) * back
            else:
                r = layer.self_attn(x, None, None)     # no mask: bidirectional
            h = h + r
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return m.norm(h)[0]

    def decide(self, state, question, options):
        mx = self.mx
        t0 = time.monotonic()
        ids, readout, ends, labels = self.encode(state, question, options)
        h = self.hidden(ids)
        r = h[readout].astype(mx.float32) @ self.query.astype(mx.float32).T          # (rank,)
        k = h[mx.array(ends)].astype(mx.float32) @ self.key.astype(mx.float32).T      # (n, rank)
        logits = (k @ r) / self.rank
        probs = mx.softmax(logits)
        mx.eval(probs)
        ms = (time.monotonic() - t0) * 1000
        return make_answer(dict(zip(labels, probs.tolist()))), {"total_ms": ms, "tokens": len(ids)}


def load(size: str = "4b", backend: str = "auto", device: str | None = None) -> _Base:
    if backend == "auto":
        try:
            import mlx.core  # noqa: F401
            import mlx_lm    # noqa: F401
            backend = "mlx"
        except ImportError:
            backend = "torch"
    return Decision2MLX(size) if backend == "mlx" else Decision2Torch(size, device)
