"""Everything that can be verified without loading a model or an API key.

    python selfcheck.py

Exists because "it's set up" should be a thing you can run, not a claim.
"""
import os, sys

import vizdoom as vzd
from dotenv import load_dotenv

load_dotenv()
from doom_agent import FuncBrain, heuristic, play  # noqa: E402

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")

def note(name, cond, detail=""):
    """Optional pieces: reported, never fail the exit code."""
    print(f"  {'PASS' if cond else 'note'}  {name}{'  ' + detail if detail else ''}")

check("vizdoom engine", hasattr(vzd, "DoomGame"), vzd.__version__)
check("scenario present",
      os.path.exists(os.path.join(vzd.scenarios_path, "defend_the_center.cfg")))

# Cribl Decision 1.0 (local)
try:
    import torch, transformers, peft  # noqa: F401
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    check("cribl deps (torch/transformers/peft)", True,
          f"torch {torch.__version__}, transformers {transformers.__version__}, peft {peft.__version__}, device {dev}")
except ImportError as e:
    check("cribl deps (torch/transformers/peft)", False, f"{e}; uv sync --extra cribl")
try:
    import mlx_lm  # noqa: F401
    note("mlx backend", True, "Apple Silicon fast path available")
except ImportError:
    note("mlx backend", False, "not installed; --backend torch will be used")
try:
    from huggingface_hub import try_to_load_from_cache
    from cribl_decision import ADAPTER_ID, ADAPTER_REVISION, BASE_ID, BASE_REVISION
    a = try_to_load_from_cache(ADAPTER_ID, "adapter_model.safetensors", revision=ADAPTER_REVISION)
    b = try_to_load_from_cache(BASE_ID, "model.safetensors-00002-of-00002.safetensors", revision=BASE_REVISION)
    note("cribl weights cached", isinstance(a, str) and isinstance(b, str),
         "adapter + pinned base present" if isinstance(a, str) and isinstance(b, str)
         else "will download ~9.4 GB on first --brain cribl run")
except Exception as e:  # noqa: BLE001
    note("cribl weights cached", False, str(e))

# Jev (optional)
try:
    from typesafe_sdk import Choice, Noul, TypeSafeClient  # noqa: F401
    note("typesafe-sdk", True)
except ImportError:
    note("typesafe-sdk", False, "not installed; --brain jev needs uv sync --extra jev")
has_key = bool(os.environ.get("TYPESAFE_API_KEY"))
note("TYPESAFE_API_KEY", has_key, "set" if has_key else "not set; only --brain jev needs it")

# The bug that cost two zeroed baselines: bearing sign. A monster to the LEFT
# has a POSITIVE bearing, so the heuristic must turn left for it.
check("turn direction", heuristic({"monsters_nearest_first":
      [{"kind": "Demon", "distance": 200, "bearing_deg": 90}]}) == "TURN_LEFT")
check("turn direction (mirror)", heuristic({"monsters_nearest_first":
      [{"kind": "Demon", "distance": 200, "bearing_deg": -90}]}) == "TURN_RIGHT")
check("fires when aligned", heuristic({"monsters_nearest_first":
      [{"kind": "Demon", "distance": 200, "bearing_deg": 2}]}) == "ATTACK")

s = play(FuncBrain(heuristic, "heuristic"), episodes=2, tics=4, quiet=True, seed=1234)
check("heuristic plays", s["kills_mean"] > 2, f"{s['kills_mean']:.1f} kills (7.55 over 20 episodes)")

print("\n" + ("setup ok" if ok else "SETUP BROKEN, see failures above"))
sys.exit(0 if ok else 1)
