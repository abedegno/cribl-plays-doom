"""Doom, played by a decision model that never sees the screen.

    python doom_agent.py --brain heuristic                  # no model needed
    python doom_agent.py --brain cribl --episodes 20 --seed 1234
    python doom_agent.py --brain jev                        # needs TYPESAFE_API_KEY

The model gets a structured description of the world (health, ammo, monster
bearings and distances) and returns one of three buttons with a probability
for each. No pixels go in and no text comes out.
"""
from __future__ import annotations

import argparse, csv, json, math, os, random, statistics, threading, time
from dataclasses import asdict, dataclass, field

import vizdoom as vzd
from dotenv import load_dotenv

# Read .env before anything touches os.environ, so TYPESAFE_API_KEY can live in
# a file instead of being exported into every shell. A real environment
# variable still wins -- load_dotenv does not override what is already set.
load_dotenv()

TICS_PER_SEC = 35
BUTTONS = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]   # order of the scenario's buttons
# Order the options are offered to a model in. Separate from BUTTONS so the
# order-sensitivity experiment can permute it without touching the game.
OPTION_ORDER = ["ATTACK", "TURN_LEFT", "TURN_RIGHT"]

# Two ways to describe the same three buttons, kept side by side because the
# difference between them is the actual experiment.
#
# RULE spells out the 8-degree threshold the hand-coded brain uses, so the model
# is asked to evaluate a condition it was handed. INTENT describes what each
# button is FOR and leaves the model to decide what "lined up" means. Only the
# second measures judgment; the first measures instruction-following.
CRITERIA_RULE = {
    "ATTACK": "Fire. Correct whenever the nearest monster's bearing is between "
              "-8 and +8, i.e. already lined up. Firing is how you score; do it "
              "the moment you are aligned rather than adjusting further.",
    "TURN_LEFT": "Rotate left. Correct only when the nearest monster's bearing is "
                 "GREATER than +8. Do not pick this if the bearing is between -8 "
                 "and +8; fire instead.",
    "TURN_RIGHT": "Rotate right. Correct only when the nearest monster's bearing "
                  "is LESS than -8. Do not pick this if the bearing is between -8 "
                  "and +8; fire instead.",
}

CRITERIA_INTENT = {
    "ATTACK": "Fire your weapon down your line of sight. This is the only way to "
              "score, but a shot that is not lined up hits nothing and wastes a "
              "bullet you cannot spare.",
    "TURN_LEFT": "Rotate left. This swings your line of sight toward monsters that "
                 "are to your left, and away from ones on your right.",
    "TURN_RIGHT": "Rotate right. This swings your line of sight toward monsters "
                  "that are to your right, and away from ones on your left.",
}

# Prompt-engineering experiment (A): the bare minimum, and a goal statement with
# no per-button tactics.
CRITERIA_MINIMAL = {"ATTACK": "Fire.", "TURN_LEFT": "Turn left.", "TURN_RIGHT": "Turn right."}
CRITERIA_GOAL = {
    "ATTACK": "Shoot straight ahead.",
    "TURN_LEFT": "Rotate your view to the left.",
    "TURN_RIGHT": "Rotate your view to the right.",
}
CRITERIA = {"rule": CRITERIA_RULE, "intent": CRITERIA_INTENT,
            "minimal": CRITERIA_MINIMAL, "goal": CRITERIA_GOAL}

INSTRUCTIONS = {
    "default": "You are fighting for your life in Doom, surrounded by monsters. "
               "Pick the single best button to press right now.",
    "goal": "You are in Doom, standing still in the centre of a circular room, "
            "and monsters walk towards you from every side. You score a point for "
            "every monster you kill and lose if they reach and kill you. You have "
            "limited bullets, and a bullet only hits a monster that is directly in "
            "front of you (bearing near 0). Pick the button that best serves "
            "killing as many monsters as possible.",
}
CRITERIA_INSTRUCTIONS = {"rule": "default", "intent": "default", "minimal": "default",
                         "goal": "goal"}

MONSTERS = ("Demon", "Cacodemon", "ZombieMan", "ShotgunGuy", "HellKnight", "Imp", "Revenant")
GAME_VARS = [vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2,
             vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE,
             # below here: scoring only, never put in the model's state
             vzd.GameVariable.ARMOR, vzd.GameVariable.KILLCOUNT, vzd.GameVariable.HITCOUNT,
             vzd.GameVariable.SELECTED_WEAPON_AMMO]
FOV_HALF_DEG = 45   # ViZDoom default 90 degree horizontal field of view


def describe(state, history: list[str] | None = None, rich: bool | str = False) -> dict:
    """The whole game as structured state.

    The default ("jev") state is byte-for-byte what the original Jev agent
    received, so scores are comparable. `rich` adds armour, the count of
    monsters inside the field of view, and the last few actions -- all derived
    from the same information, nothing the Jev agent could not have computed.
    """
    hp, ammo, px, py, ang = state.game_variables[:5]
    monsters = []
    for o in state.objects:
        # MarineChainsawVzd is decoration standing around the arena, not a
        # threat. Leaving it in made both brains aim at furniture.
        if o.name in MONSTERS:
            dx, dy = o.position_x - px, o.position_y - py
            dist = math.hypot(dx, dy)
            # bearing relative to where the player is facing: 0 = dead ahead,
            # positive = to the left. This is the number the decision turns on.
            bearing = (math.degrees(math.atan2(dy, dx)) - ang + 540) % 360 - 180
            monsters.append({"kind": o.name, "distance": round(dist),
                             "bearing_deg": round(bearing)})
    # Nearest by DISTANCE, not by bearing: sorting on bearing made the target
    # flip between monsters on either side as the player turned, so it spun in
    # place instead of converging on anything.
    monsters.sort(key=lambda m: m["distance"])
    d = {
        "health": int(hp), "ammo": int(ammo),
        "facing_deg": round(ang),
        "monsters_nearest_first": monsters[:6],
        "note": "bearing 0 = dead ahead. POSITIVE bearing means the monster is to "
                "your LEFT (turn left to face it); NEGATIVE means to your RIGHT.",
    }
    if rich == "compact":
        near = monsters[0] if monsters else None
        return {
            "health": int(hp), "ammo": int(ammo), "armor": int(state.game_variables[5]),
            "nearest_monster": None if near is None else
                {"distance": near["distance"], "bearing_deg": near["bearing_deg"]},
            "monsters_in_view": sum(abs(m["bearing_deg"]) <= FOV_HALF_DEG for m in monsters),
            "monsters_total": len(monsters),
            "previous_action": (history or [None])[-1],
            "note": d["note"],
        }
    if rich:
        d["armor"] = int(state.game_variables[5])
        d["monsters_in_view"] = sum(abs(m["bearing_deg"]) <= FOV_HALF_DEG for m in monsters)
        d["previous_actions"] = list(history or [])[-3:]
    return d


@dataclass
class Decision:
    action: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None      # official system-one concentration formula
    p_selected: float | None = None      # probability of the chosen action
    model_ms: float = 0.0                # forward pass + readout
    fallback: str | None = None          # None, or why the brain did not decide itself
    model_action: str | None = None      # what the model picked, when overridden
    extra: dict = field(default_factory=dict)


def heuristic(desc: dict) -> str:
    """A bar to beat, so the model's score means something."""
    ms = desc.get("monsters_nearest_first")
    if ms is None:                                  # compact state
        ms = [desc["nearest_monster"]] if desc.get("nearest_monster") else []
    if not ms:
        return "TURN_RIGHT"
    b = ms[0]["bearing_deg"]
    if abs(b) <= 8:
        return "ATTACK"
    return "TURN_LEFT" if b > 0 else "TURN_RIGHT"


class FuncBrain:
    """Wrap a plain `desc -> button` function (the original brain interface)."""
    def __init__(self, fn, name):
        self.fn, self.name = fn, name
    def decide(self, desc: dict) -> Decision:
        return Decision(self.fn(desc))
    def __call__(self, desc: dict) -> str:
        return self.decide(desc).action


class RandomBrain(FuncBrain):
    def __init__(self, seed: int | None):
        rng = random.Random(seed)
        super().__init__(lambda d: rng.choice(BUTTONS), "random")


class JevBrain:
    """TypeSafe's hosted Jev, unchanged from the original project apart from
    returning a Decision. Needs TYPESAFE_API_KEY."""
    name = "jev"

    def __init__(self, model: str = "jev-latest", conf_floor: float = 0.0,
                 criteria: str = "rule"):
        from typesafe_sdk import TypeSafeClient
        self.client = TypeSafeClient(model=model)
        self.conf_floor = conf_floor
        self.criteria = CRITERIA[criteria]
        self.instructions = INSTRUCTIONS[CRITERIA_INSTRUCTIONS[criteria]]

    def decide(self, desc: dict) -> Decision:
        from typesafe_sdk import Choice, Noul
        t0 = time.monotonic()
        r = self.client.system_one(
            state=desc,
            questions={
                "action": Choice(instructions=self.instructions, criteria=self.criteria),
                "in_danger": Noul(instructions="Is a monster close enough to hurt you "
                                               "within the next second?"),
            },
        )
        a = r.choices["action"]
        d = Decision(str(a.choice), dict(getattr(a, "probabilities", {}) or {}),
                     a.confidence, None, (time.monotonic() - t0) * 1000)
        if d.probabilities:
            d.p_selected = d.probabilities.get(d.action)
        if a.confidence < self.conf_floor:
            d.model_action, d.action, d.fallback = d.action, heuristic(desc), "low_confidence"
        return d

    def __call__(self, desc: dict) -> str:
        return self.decide(desc).action


def load_cribl_model(backend: str = "auto", fmt: str = "plain_letter_qfirst", **kw):
    """mlx on Apple Silicon when mlx-lm is installed, else PyTorch (CUDA,
    MPS or CPU). Both run the same prompt and readout; see cribl_mlx.py."""
    if backend == "auto":
        try:
            import mlx.core  # noqa: F401
            import mlx_lm    # noqa: F401
            backend = "mlx"
        except ImportError:
            backend = "torch"
    if backend == "mlx":
        from cribl_mlx import CriblDecisionModelMLX
        return CriblDecisionModelMLX(fmt=fmt)
    from cribl_decision import CriblDecisionModel
    return CriblDecisionModel(fmt=fmt, **kw)


class CriblDecisionBrain:
    """cribl-ai/cribl-decision-1.0 run locally (see cribl_decision.py).

    The model is loaded once and reused for every decision. One typed `choice`
    question over the three buttons; the answer is the argmax of the
    candidate distribution read from the native LM head.

    Fallbacks are never silent. `on_error` decides what happens when the model
    raises: "raise" (default) stops the run, "heuristic"/"noop" substitute and
    count it. `conf_floor` substitutes the heuristic for low-confidence
    decisions, again counted and traced with the model's own pick.
    """
    name = "cribl"

    def __init__(self, model=None, criteria: str = "rule", fmt: str = "plain_letter_qfirst",
                 option_order: list[str] | None = None, conf_floor: float = 0.0,
                 on_error: str = "raise", ask_danger: bool = False, backend: str = "auto",
                 **model_kw):
        from cribl_decision import Question
        self.model = model or load_cribl_model(backend, fmt, **model_kw)
        if model is not None:
            from cribl_decision import FORMATS
            self.model.fmt = FORMATS[fmt]
        crit = CRITERIA[criteria]
        order = option_order or OPTION_ORDER
        self.question = Question("choice", INSTRUCTIONS[CRITERIA_INSTRUCTIONS[criteria]],
                                 {k: crit[k] for k in order})
        self.danger = Question("noul", "Is a monster close enough to hurt you within "
                                       "the next second?") if ask_danger else None
        self.conf_floor, self.on_error = conf_floor, on_error

    def decide(self, desc: dict) -> Decision:
        qs = {"action": self.question}
        if self.danger is not None:
            qs["in_danger"] = self.danger
        try:
            answers, timing = self.model.ask(desc, qs)
        except Exception as e:                       # noqa: BLE001
            if self.on_error == "raise":
                raise
            sub = heuristic(desc) if self.on_error == "heuristic" else "TURN_RIGHT"
            return Decision(sub, fallback=f"error:{type(e).__name__}")
        a = answers["action"]
        d = Decision(a.choice, a.probabilities, a.confidence, a.p_selected,
                     timing["total_ms"], extra={"tokens": timing["tokens"],
                                                "candidate_mass": a.candidate_mass})
        if "in_danger" in answers:
            d.extra["in_danger"] = answers["in_danger"].noul
        if a.confidence < self.conf_floor:
            d.model_action, d.action, d.fallback = a.choice, heuristic(desc), "low_confidence"
        return d

    def __call__(self, desc: dict) -> str:
        return self.decide(desc).action


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def make_game(render: bool = False, window: bool = False) -> vzd.DoomGame:
    g = vzd.DoomGame()
    g.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    g.set_window_visible(window)
    g.set_objects_info_enabled(True)
    g.set_available_game_variables(GAME_VARS)
    if render:
        g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_render_hud(True)
    g.init()
    return g


@dataclass
class EpisodeResult:
    episode: int
    seed: int | None
    reward: float          # the scenario's score: +1 per kill, -1 for dying
    kills: int
    died: bool
    survival_tics: int
    shots: int
    hits: int
    decisions: int
    fallbacks: int
    wall_s: float
    model_ms: list = field(default_factory=list, repr=False)
    decision_ms: list = field(default_factory=list, repr=False)
    p_selected: list = field(default_factory=list, repr=False)
    confidence: list = field(default_factory=list, repr=False)

    @property
    def accuracy(self) -> float | None:
        return self.hits / self.shots if self.shots else None


class Tracer:
    """One CSV row per decision (and optionally per tic in live mode)."""
    FIELDS = ["brain", "episode", "tic", "health", "ammo", "kills", "n_monsters",
              "nearest_bearing", "nearest_distance", "action", "model_action",
              "fallback", "p_ATTACK", "p_TURN_LEFT", "p_TURN_RIGHT", "p_selected",
              "confidence", "model_ms", "decision_ms", "stale_tics", "state"]

    def __init__(self, path: str | None, brain: str):
        self.brain = brain
        self.f = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.f = open(path, "w", newline="")
            self.w = csv.DictWriter(self.f, self.FIELDS)
            self.w.writeheader()

    def row(self, ep, tic, desc, kills, d: Decision, decision_ms, stale_tics=0):
        if not self.f:
            return
        ms = desc.get("monsters_nearest_first")
        if ms is None:
            ms = [desc["nearest_monster"]] if desc.get("nearest_monster") else []
        p = d.probabilities or {}
        self.w.writerow({
            "brain": self.brain, "episode": ep, "tic": tic, "health": desc["health"],
            "ammo": desc["ammo"], "kills": kills, "n_monsters": len(ms),
            "nearest_bearing": ms[0]["bearing_deg"] if ms else "",
            "nearest_distance": ms[0]["distance"] if ms else "",
            "action": d.action, "model_action": d.model_action or "", "fallback": d.fallback or "",
            **{f"p_{k}": round(p[k], 5) if k in p else "" for k in BUTTONS},
            "p_selected": "" if d.p_selected is None else round(d.p_selected, 5),
            "confidence": "" if d.confidence is None else round(d.confidence, 5),
            "model_ms": round(d.model_ms, 2), "decision_ms": round(decision_ms, 2),
            "stale_tics": stale_tics, "state": json.dumps(desc, separators=(",", ":"))})

    def close(self):
        if self.f:
            self.f.close()


def _decide(brain, desc) -> Decision:
    if hasattr(brain, "decide"):
        return brain.decide(desc)
    return Decision(brain(desc))        # plain callable, original interface


def run_episode(g, brain, ep: int, seed: int | None, tics: int, rich: bool,
                tracer: Tracer | None = None, on_tic=None) -> EpisodeResult:
    """Lock-step: the game waits for every decision, so game-time results do
    not depend on how slow the brain is. Wall-clock cost is measured apart."""
    if seed is not None:
        g.set_seed(seed)          # same start for every brain, every episode
    g.new_episode()
    ammo0 = int(g.get_game_variable(vzd.GameVariable.AMMO2))
    history: list[str] = []
    r = EpisodeResult(ep, seed, 0, 0, False, 0, 0, 0, 0, 0, 0.0)
    w0 = time.monotonic()
    while not g.is_episode_finished():
        s = g.get_state()
        if s is None:
            break
        desc = describe(s, history, rich)
        t0 = time.monotonic()
        d = _decide(brain, desc)
        dt = (time.monotonic() - t0) * 1000
        if d.action not in BUTTONS:
            raise ValueError(f"brain returned {d.action!r}, not one of {BUTTONS}")
        r.decisions += 1; r.decision_ms.append(dt)
        if d.model_ms: r.model_ms.append(d.model_ms)
        if d.p_selected is not None: r.p_selected.append(d.p_selected)
        if d.confidence is not None: r.confidence.append(d.confidence)
        if d.fallback: r.fallbacks += 1
        if tracer:
            tracer.row(ep, s.tic, desc, int(s.game_variables[6]), d, dt)
        history.append(d.action)
        act = [1 if b == d.action else 0 for b in BUTTONS]
        if on_tic is None:
            g.make_action(act, tics)
        else:
            for _ in range(tics):            # tic by tic so a recorder sees every frame
                if g.is_episode_finished():
                    break
                g.make_action(act, 1)
                on_tic(g, d, desc)
    r.wall_s = time.monotonic() - w0
    r.reward = g.get_total_reward()
    r.kills = int(g.get_game_variable(vzd.GameVariable.KILLCOUNT))
    r.hits = int(g.get_game_variable(vzd.GameVariable.HITCOUNT))
    r.shots = ammo0 - int(g.get_game_variable(vzd.GameVariable.AMMO2))
    r.died = g.is_player_dead()
    r.survival_tics = g.get_episode_time() - g.get_episode_start_time()
    return r


def run_episode_live(g, brain, ep: int, seed: int | None, tics: int, rich: bool,
                     tracer: Tracer | None = None, on_tic=None) -> EpisodeResult:
    """Real-time: the game advances at 35 tics/s of wall clock whether or not
    the model has answered. A worker thread takes the latest state snapshot,
    decides, and posts the action; the game applies it for `tics` tics and then
    idles (no buttons) until the next decision lands. The model therefore only
    ever sees state that is current or stale -- never future state -- and the
    staleness (tics elapsed during inference) is traced."""
    if seed is not None:
        g.set_seed(seed)
    g.new_episode()
    ammo0 = int(g.get_game_variable(vzd.GameVariable.AMMO2))
    r = EpisodeResult(ep, seed, 0, 0, False, 0, 0, 0, 0, 0, 0.0)
    lock = threading.Lock()
    snapshot = {"desc": None, "tic": 0}
    pending: list = []                      # decisions ready to apply
    stop = threading.Event()
    history: list[str] = []

    def worker():
        while not stop.is_set():
            with lock:
                desc, tic0 = snapshot["desc"], snapshot["tic"]
            if desc is None:
                time.sleep(0.001); continue
            t0 = time.monotonic()
            try:
                d = _decide(brain, desc)
            except Exception as e:             # surfaced, not swallowed
                d = Decision("TURN_RIGHT", fallback=f"error:{type(e).__name__}")
            dt = (time.monotonic() - t0) * 1000
            with lock:
                pending.append((d, desc, tic0, dt))
                snapshot["desc"] = None        # wait for a fresh state

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    period = 1.0 / TICS_PER_SEC
    w0 = next_t = time.monotonic()
    current, left = None, 0
    try:
        while not g.is_episode_finished():
            s = g.get_state()
            if s is None:
                break
            with lock:
                if snapshot["desc"] is None and not pending:
                    snapshot["desc"] = describe(s, history, rich); snapshot["tic"] = s.tic
                got = pending.pop() if pending else None
            if got:
                d, desc, tic0, dt = got
                d.extra["stale_tics"] = s.tic - tic0
                r.decisions += 1; r.decision_ms.append(dt)
                if d.model_ms: r.model_ms.append(d.model_ms)
                if d.p_selected is not None: r.p_selected.append(d.p_selected)
                if d.confidence is not None: r.confidence.append(d.confidence)
                if d.fallback: r.fallbacks += 1
                if tracer:
                    tracer.row(ep, tic0, desc, int(s.game_variables[6]), d, dt,
                               stale_tics=s.tic - tic0)
                history.append(d.action)
                current, left = d, tics
            act = [1 if (current and left > 0 and b == current.action) else 0 for b in BUTTONS]
            g.make_action(act, 1)
            left -= 1
            if on_tic is not None and current is not None:
                on_tic(g, current, None)
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    finally:
        stop.set(); th.join(timeout=5)
    r.wall_s = time.monotonic() - w0
    r.reward = g.get_total_reward()
    r.kills = int(g.get_game_variable(vzd.GameVariable.KILLCOUNT))
    r.hits = int(g.get_game_variable(vzd.GameVariable.HITCOUNT))
    r.shots = ammo0 - int(g.get_game_variable(vzd.GameVariable.AMMO2))
    r.died = g.is_player_dead()
    r.survival_tics = g.get_episode_time() - g.get_episode_start_time()
    return r


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarise(results: list[EpisodeResult], tics: int) -> dict:
    def mean(xs): return statistics.mean(xs) if xs else None
    dm = [x for r in results for x in r.decision_ms]
    mm = [x for r in results for x in r.model_ms]
    ps = [x for r in results for x in r.p_selected]
    cs = [x for r in results for x in r.confidence]
    shots = sum(r.shots for r in results); hits = sum(r.hits for r in results)
    decisions = sum(r.decisions for r in results)
    wall = sum(r.wall_s for r in results)
    game_s = sum(r.survival_tics for r in results) / TICS_PER_SEC
    return {
        "episodes": len(results), "tics_per_decision": tics,
        "reward_mean": mean([r.reward for r in results]),
        "reward_min": min(r.reward for r in results), "reward_max": max(r.reward for r in results),
        "kills_mean": mean([r.kills for r in results]),
        "kills_sd": statistics.stdev([r.kills for r in results]) if len(results) > 1 else 0.0,
        "deaths": sum(r.died for r in results),
        "survival_tics_mean": mean([r.survival_tics for r in results]),
        "survival_s_mean": mean([r.survival_tics / TICS_PER_SEC for r in results]),
        "shots": shots, "hits": hits, "accuracy": hits / shots if shots else None,
        "decisions": decisions, "fallbacks": sum(r.fallbacks for r in results),
        "decision_ms_mean": mean(dm), "decision_ms_p50": pct(dm, .5), "decision_ms_p95": pct(dm, .95),
        "model_ms_mean": mean(mm), "model_ms_p50": pct(mm, .5), "model_ms_p95": pct(mm, .95),
        "p_selected_mean": mean(ps), "confidence_mean": mean(cs),
        "wall_s": wall, "game_s": game_s,
        "decisions_per_wall_s": decisions / wall if wall else None,
        "game_s_per_wall_s": game_s / wall if wall else None,
        "realtime_decision_budget_ms": tics / TICS_PER_SEC * 1000,
        "per_episode": [{k: v for k, v in asdict(r).items()
                         if k not in ("model_ms", "decision_ms", "p_selected", "confidence")}
                        | {"accuracy": r.accuracy} for r in results],
    }


def warmup(brain, g):
    """One untimed decision before the first episode, so kernel compilation and
    the prefix cache build are not charged to episode 1's first decision."""
    if not hasattr(brain, "model"):
        return
    g.set_seed(0); g.new_episode()
    s = g.get_state()
    if s is not None:
        brain.decide(describe(s))


def play(brain, episodes: int, tics: int, quiet: bool, seed: int | None,
         rich: bool = False, trace: str | None = None, live: bool = False,
         name: str = "brain", window: bool = False) -> dict:
    g = make_game(window=window)
    warmup(brain, g)
    tracer = Tracer(trace, name)
    results = []
    try:
        for ep in range(episodes):
            ep_seed = None if seed is None else seed + ep
            fn = run_episode_live if live else run_episode
            r = fn(g, brain, ep + 1, ep_seed, tics, rich, tracer)
            results.append(r)
            if not quiet:
                lat = f", {statistics.mean(r.decision_ms):.0f} ms/decision" if r.decision_ms else ""
                print(f"  episode {ep+1:2d} (seed {ep_seed}): {r.kills} kills, reward {r.reward:.0f}, "
                      f"{'died' if r.died else 'survived'} at {r.survival_tics/TICS_PER_SEC:.1f}s, "
                      f"hits {r.hits}/{r.shots}, {r.decisions} decisions{lat}"
                      + (f", {r.fallbacks} fallbacks" if r.fallbacks else ""), flush=True)
    finally:
        tracer.close(); g.close()
    return summarise(results, tics)


def build_brain(a, model=None):
    if a.brain == "heuristic":
        return FuncBrain(heuristic, "heuristic")
    if a.brain == "random":
        return RandomBrain(None if a.seed < 0 else a.seed)
    if a.brain == "jev":
        return JevBrain(a.jev_model, a.conf_floor, a.criteria)
    order = a.order.split(",") if a.order else None
    return CriblDecisionBrain(model=model, criteria=a.criteria, fmt=a.format,
                              option_order=order, conf_floor=a.conf_floor,
                              on_error=a.on_error, ask_danger=a.ask_danger,
                              backend=a.backend, device=a.device, dtype=a.dtype)


def state_mode(a):
    return {"jev": False, "rich": True, "compact": "compact"}[getattr(a, "state", "jev")]


def add_args(ap: argparse.ArgumentParser):
    ap.add_argument("--brain", choices=["heuristic", "random", "jev", "cribl"], default="heuristic")
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--conf-floor", type=float, default=0.0,
                    help="below this confidence, substitute the heuristic (counted as a fallback)")
    ap.add_argument("--criteria", choices=list(CRITERIA), default="rule",
                    help="rule: hand the model the 8-degree threshold. intent: describe "
                         "what each button is for. minimal: names only. goal: goal-oriented "
                         "instructions with plain button descriptions.")
    ap.add_argument("--backend", choices=["auto", "mlx", "torch"], default="auto",
                    help="cribl inference backend: mlx (Apple Silicon) or torch (CUDA/MPS/CPU)")
    ap.add_argument("--format", default="plain_letter_qfirst", help="cribl prompt format (cribl_decision.FORMATS)")
    ap.add_argument("--order", default="", help="cribl option order, e.g. TURN_RIGHT,ATTACK,TURN_LEFT")
    ap.add_argument("--on-error", choices=["raise", "heuristic", "noop"], default="raise")
    ap.add_argument("--ask-danger", action="store_true",
                    help="also ask the noul 'in danger?' question in the same forward batch")
    ap.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--state", choices=["jev", "rich", "compact"], default="jev",
                    help="jev: exactly the original Jev agent's state. rich: + armour, monsters in "
                         "view, last actions. compact: nearest monster first-class, no list")
    ap.add_argument("--live", action="store_true",
                    help="real-time: the game runs at 35 tics/s while the model thinks")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--tics", type=int, default=4, help="game tics per decision")
    ap.add_argument("--seed", type=int, default=1234,
                    help="episode i uses seed+i, identical for every brain; -1 randomises")


def main() -> int:
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--trace", default="", help="CSV of every decision")
    ap.add_argument("--json", default="", help="write the summary here")
    ap.add_argument("--window", action="store_true", help="show the ViZDoom window")
    a = ap.parse_args()

    brain = build_brain(a)
    seed = None if a.seed < 0 else a.seed
    tag = f" · criteria={a.criteria}" if a.brain in ("jev", "cribl") else ""
    if a.brain == "cribl":
        m = brain.model
        tag += (f" · format={a.format} · {m.device}/{getattr(m, 'dtype_name', a.dtype)}"
                f" · loaded in {m.load_seconds:.1f}s"
                + (f" · LoRA merged into {m.adapter_report['modules_merged']} modules"
                   if "modules_merged" in m.adapter_report else
                   f" · adapter tensors verified {m.adapter_report['tensors']}"))
    print(f"defend_the_center · brain={a.brain}{tag} · {a.episodes} episodes · seed {seed}"
          f" · {a.tics} tics/decision{' · LIVE' if a.live else ''}")
    s = play(brain, a.episodes, a.tics, a.quiet, seed, state_mode(a), a.trace or None,
             a.live, a.brain, a.window)
    print(f"\n  {s['decisions']} decisions, {s['fallbacks']} fallbacks")
    if s["decision_ms_mean"] is not None and a.brain != "heuristic":
        print(f"  decision latency mean {s['decision_ms_mean']:.1f} ms, p50 {s['decision_ms_p50']:.1f}, "
              f"p95 {s['decision_ms_p95']:.1f}  (real-time budget {s['realtime_decision_budget_ms']:.0f} ms)")
    if s["p_selected_mean"] is not None:
        print(f"  selected-action probability mean {s['p_selected_mean']:.3f}, "
              f"confidence mean {s['confidence_mean']:.3f}")
    print(f"  wall {s['wall_s']:.1f}s for {s['game_s']:.1f}s of game time "
          f"({s['game_s_per_wall_s']:.2f}x real time), {s['decisions_per_wall_s']:.1f} decisions/s")
    acc = f"{s['accuracy']:.2f}" if s["accuracy"] is not None else "n/a"
    print(f"  MEAN {s['kills_mean']:.2f} kills (reward {s['reward_mean']:.2f}, min {s['reward_min']:.0f}, "
          f"max {s['reward_max']:.0f}) · {s['deaths']}/{s['episodes']} died · "
          f"survival {s['survival_s_mean']:.1f}s · accuracy {acc}")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"args": vars(a), "summary": s}, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
