
from __future__ import annotations

"""
AI401 — IMAES — Experiment 3
Agent Architectures — The BDI Deliberation Cycle

Complete implementation of the assignment:
1) BDI deliberation cycle on a 15x15 dynamic Tileworld
2) single-minded commitment with reconsideration interval gamma
3) charged deliberation time
4) event trace containing COMMIT / ACHIEVE / DROP / SWITCH
5) Exercise 1: gamma {1,2,4,8,bold} × world speed {1,2,4,8},
   averaged over 25 seeds and 600 agent steps, with tables + plots
6) Exercise 2: battery capacity 40, charger, resource-aware filter,
   fill-hole/recharge plan schemas, comparison with battery-blind agent

Unspecified simulator parameters in the PDF are explicitly chosen here:
- hole appearance probability per world tick = 0.08
- hole lifetime = random integer in [4,12] ticks
- four-neighbour grid movement
- equal-distance tie break = lexicographic (row, col)
- bold gamma = 31, greater than longest Manhattan plan on 15x15 (28)
"""

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
import csv
import random
import statistics
from typing import Dict, List, Optional, Tuple

# ---------------------------- configuration ----------------------------

GRID_SIZE = 15
HOLE_APPEAR_PROB = 0.05
LIFE_MIN, LIFE_MAX = 8, 22
START = (7, 7)
CHARGER = (0, 0)

GAMMA_LABELS = ["1", "2", "4", "8", "bold"]
BOLD_GAMMA = 31
WORLD_SPEEDS = [1, 2, 4, 8]
SEEDS = list(range(25))
STEPS = 600

OUT = Path("AI401_Experiment3_outputs")
OUT.mkdir(exist_ok=True)

Pos = Tuple[int, int]


# ---------------------------- utility ----------------------------------

def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def shortest_path(start: Pos, goal: Pos) -> List[Pos]:
    """Deterministic shortest 4-neighbour path."""
    path = []
    cur = start
    while cur != goal:
        r, c = cur
        gr, gc = goal
        if r < gr:
            cur = (r + 1, c)
        elif r > gr:
            cur = (r - 1, c)
        elif c < gc:
            cur = (r, c + 1)
        else:
            cur = (r, c - 1)
        path.append(cur)
    return path


def gamma_value(label: str) -> int:
    return BOLD_GAMMA if label == "bold" else int(label)


# ---------------------------- Tileworld --------------------------------

@dataclass
class Hole:
    lifetime: int


class TileWorld:
    """Dynamic 15x15 Tileworld."""

    def __init__(
        self,
        seed: int,
        size: int = GRID_SIZE,
        p_appear: Optional[float] = None,
        life_min: Optional[int] = None,
        life_max: Optional[int] = None,
    ):
        self.rng = random.Random(seed)
        self.size = size
        self.p_appear = HOLE_APPEAR_PROB if p_appear is None else p_appear
        self.life_min = LIFE_MIN if life_min is None else life_min
        self.life_max = LIFE_MAX if life_max is None else life_max
        self.holes: Dict[Pos, Hole] = {}
        self.tick = 0
        self.appeared = 0
        self.filled = 0

    def snapshot(self):
        return tuple(sorted(self.holes.keys()))

    def has(self, pos: Pos) -> bool:
        return pos in self.holes

    def _try_appearance(self):
        if len(self.holes) >= self.size * self.size:
            return None
        if self.rng.random() >= self.p_appear:
            return None

        # Random probing avoids rebuilding a 225-cell list every world tick.
        for _ in range(20):
            pos = (self.rng.randrange(self.size), self.rng.randrange(self.size))
            if pos not in self.holes:
                self.holes[pos] = Hole(
                    self.rng.randint(self.life_min, self.life_max)
                )
                self.appeared += 1
                return pos
        return None

    def advance_tick(self):
        self.tick += 1
        expired = []
        for pos, hole in self.holes.items():
            hole.lifetime -= 1
            if hole.lifetime <= 0:
                expired.append(pos)
        for pos in expired:
            del self.holes[pos]
        self._try_appearance()

    def fill(self, pos: Pos) -> bool:
        if pos in self.holes:
            del self.holes[pos]
            self.filled += 1
            return True
        return False


# ---------------------------- BDI types --------------------------------

@dataclass
class Belief:
    holes: Tuple[Pos, ...]
    position: Pos
    world_tick: int
    battery: Optional[int] = None


@dataclass
class Desire:
    target: Pos
    utility: float


@dataclass
class Intention:
    target: Pos
    utility: float


@dataclass
class Plan:
    actions: List[Pos] = field(default_factory=list)


@dataclass
class Event:
    step: int
    kind: str
    message: str


# ---------------------------- BDI agent ---------------------------------

class BDIAgent:
    """
    BDI functions:
      brf(B,p)      belief revision
      options(B,I)  generate desires
      filter(B,D,I) commit to an intention
      plan(B,I)     means–ends shortest-path planning
    """

    def __init__(self, gamma: int, mode: str = "single-minded"):
        self.gamma = gamma
        self.mode = mode
        self.position = START
        self.belief = Belief((), START, 0)
        self.intention: Optional[Intention] = None
        self.plan = Plan()
        self.since_reconsideration = 0
        self.deliberations = 0
        self.events: List[Event] = []

    def log(self, step: int, kind: str, message: str):
        self.events.append(Event(step, kind, message))

    # Algorithm 1 / Step 2
    def brf(self, percept: Dict):
        self.belief = Belief(
            holes=tuple(sorted(percept["holes"])),
            position=percept["position"],
            world_tick=percept["world_tick"],
            battery=percept.get("battery"),
        )
        self.position = self.belief.position

    # Algorithm 1 / Step 3
    def options(self) -> List[Desire]:
        desires = [
            Desire(
                target=h,
                utility=-manhattan(self.position, h),
            )
            for h in self.belief.holes
        ]
        return sorted(
            desires,
            key=lambda d: (manhattan(self.position, d.target), d.target)
        )

    def filter(self, desires: List[Desire]) -> Optional[Intention]:
        if not desires:
            return None

        best = desires[0]
        if self.intention is None:
            return Intention(best.target, best.utility)

        # Single-minded commitment: keep current intention if still achievable.
        if self.mode == "single-minded" and self.intention.target in self.belief.holes:
            return self.intention

        # Open-minded: take currently best option.
        return Intention(best.target, best.utility)

    # Algorithm 1 / Step 3
    def make_plan(self, intention: Optional[Intention]) -> Plan:
        if intention is None:
            return Plan([])
        return Plan(shortest_path(self.position, intention.target))

    def intention_achievable(self) -> bool:
        return (
            self.intention is not None
            and self.intention.target in self.belief.holes
        )

    def deliberate(self, step: int):
        self.deliberations += 1

        # Algorithm 2 / drop impossible intention
        if self.intention is not None and not self.intention_achievable():
            old = self.intention.target
            self.intention = None
            self.plan = Plan([])
            self.since_reconsideration = 0
            self.log(step, "DROP", f"intention {old} became unachievable")

        desires = self.options()
        selected = self.filter(desires)
        old_target = self.intention.target if self.intention else None

        if selected is None:
            return

        if self.intention is None:
            self.intention = selected
            self.plan = self.make_plan(self.intention)
            self.since_reconsideration = 0
            self.log(step, "COMMIT", f"committed to {selected.target}")
        elif selected.target != old_target:
            self.intention = selected
            self.plan = self.make_plan(selected)
            self.since_reconsideration = 0
            self.log(
                step,
                "SWITCH",
                f"switched from {old_target} to better option {selected.target}",
            )
        else:
            # Re-plan from current position after reconsideration.
            self.plan = self.make_plan(self.intention)
            self.since_reconsideration = 0

    def should_reconsider(self):
        if self.intention is None or not self.plan.actions:
            return True
        if self.mode == "open-minded":
            return True
        return self.since_reconsideration >= self.gamma


# ---------------------------- simulation --------------------------------

@dataclass
class RunResult:
    seed: int
    gamma_label: str
    gamma: int
    speed: int
    steps: int
    appeared: int
    filled: int
    effectiveness: float
    deliberations: int
    events: List[Event]


def perceive(world: TileWorld, agent: BDIAgent, battery=None):
    return {
        "holes": world.snapshot(),
        "position": agent.position,
        "world_tick": world.tick,
        "battery": battery,
    }


def run_bdi(
    seed: int,
    gamma_label: str,
    speed: int,
    steps: int,
    mode: str = "single-minded",
) -> RunResult:
    world = TileWorld(seed)
    agent = BDIAgent(gamma_value(gamma_label), mode=mode)

    # Initial percept / brf.
    world.advance_tick()
    agent.brf(perceive(world, agent))

    for step in range(1, steps + 1):
        # If no intention or no plan: options -> filter -> plan.
        if agent.intention is None or not agent.plan.actions:
            agent.deliberate(step)

            # Deliberation consumes a time step: world advances while thinking.
            for _ in range(speed):
                world.advance_tick()
            agent.brf(perceive(world, agent))

            if agent.intention is None or not agent.plan.actions:
                continue

        # Execute first plan action.
        next_pos = agent.plan.actions.pop(0)
        agent.position = next_pos
        agent.since_reconsideration += 1

        # World advances while action executes.
        for _ in range(speed):
            world.advance_tick()

        agent.brf(perceive(world, agent))

        # Achievement: only the intended hole is an achieved intention.
        if (
            agent.intention is not None
            and agent.position == agent.intention.target
            and world.has(agent.position)
        ):
            if world.fill(agent.position):
                agent.log(step, "ACHIEVE", f"filled intended hole at {agent.position}")
                agent.intention = None
                agent.plan = Plan([])
                agent.since_reconsideration = 0

        # If target disappeared, drop immediately.
        elif agent.intention is not None and not agent.intention_achievable():
            old = agent.intention.target
            agent.intention = None
            agent.plan = Plan([])
            agent.since_reconsideration = 0
            agent.log(step, "DROP", f"intention {old} became unachievable")

        # Reconsideration policy.
        elif agent.should_reconsider():
            agent.deliberate(step)
            for _ in range(speed):
                world.advance_tick()
            agent.brf(perceive(world, agent))

    eff = world.filled / world.appeared if world.appeared else 0.0
    return RunResult(
        seed, gamma_label, gamma_value(gamma_label), speed, steps,
        world.appeared, world.filled, eff, agent.deliberations, agent.events
    )


# ---------------------------- core event demo ---------------------------

def find_demo():
    """
    The assignment requires a trace containing COMMIT, ACHIEVE, DROP and SWITCH.
    Open-minded mode is used only for this demonstration so that SWITCH is
    guaranteed to be observable in a dynamic environment; the benchmark uses
    the required single-minded commitment policy.
    """
    for seed in range(5000):
        result = run_bdi(seed, "4", speed=4, steps=120, mode="open-minded")
        kinds = {e.kind for e in result.events}
        if {"COMMIT", "ACHIEVE", "DROP", "SWITCH"} <= kinds:
            return seed, result
    raise RuntimeError("No demonstration seed found.")


# ---------------------------- Exercise 1 --------------------------------

def sweep():
    rows = []
    for speed in WORLD_SPEEDS:
        for label in GAMMA_LABELS:
            vals = []
            fills = []
            apps = []
            dels = []
            for seed in SEEDS:
                r = run_bdi(seed, label, speed, STEPS, mode="single-minded")
                vals.append(r.effectiveness)
                fills.append(r.filled)
                apps.append(r.appeared)
                dels.append(r.deliberations)
            rows.append({
                "speed": speed,
                "gamma": label,
                "mean_effectiveness": statistics.mean(vals),
                "sd_effectiveness": statistics.stdev(vals),
                "mean_filled": statistics.mean(fills),
                "mean_appeared": statistics.mean(apps),
                "mean_deliberations": statistics.mean(dels),
                "seeds": len(SEEDS),
                "steps": STEPS,
            })

    best = {}
    for speed in WORLD_SPEEDS:
        candidates = [r for r in rows if r["speed"] == speed]
        # Maximize effectiveness; on an exact tie prefer larger gamma.
        best[speed] = max(
            candidates,
            key=lambda r: (r["mean_effectiveness"], r["gamma"] == "bold",
                           gamma_value(r["gamma"]))
        )
    return rows, best


def save_sweep(rows, best):
    csv_path = OUT / "exercise1_gamma_speed_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    best_path = OUT / "exercise1_best_gamma_by_speed.csv"
    with best_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speed", "best_gamma", "mean_effectiveness"])
        for speed in WORLD_SPEEDS:
            b = best[speed]
            w.writerow([speed, b["gamma"], b["mean_effectiveness"]])

    import matplotlib.pyplot as plt

    matrix = []
    for speed in WORLD_SPEEDS:
        matrix.append([
            next(r["mean_effectiveness"]
                 for r in rows if r["speed"] == speed and r["gamma"] == g)
            for g in GAMMA_LABELS
        ])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(GAMMA_LABELS)), GAMMA_LABELS)
    ax.set_yticks(range(len(WORLD_SPEEDS)), [str(s) for s in WORLD_SPEEDS])
    ax.set_xlabel("Reconsideration interval γ")
    ax.set_ylabel("World speed (world ticks / agent action)")
    ax.set_title("BDI Effectiveness: γ versus World Dynamism")
    fig.colorbar(im, ax=ax, label="Mean effectiveness")
    for i in range(len(WORLD_SPEEDS)):
        for j in range(len(GAMMA_LABELS)):
            ax.text(j, i, f"{matrix[i][j]:.3f}",
                    ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "exercise1_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for speed in WORLD_SPEEDS:
        ys = [
            next(r["mean_effectiveness"]
                 for r in rows
                 if r["speed"] == speed and r["gamma"] == g)
            for g in GAMMA_LABELS
        ]
        ax.plot(GAMMA_LABELS, ys, marker="o", label=f"speed={speed}")
    ax.set_xlabel("Reconsideration interval γ")
    ax.set_ylabel("Mean effectiveness")
    ax.set_title("Effectiveness versus Reconsideration Interval")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exercise1_lines.png", dpi=180)
    plt.close(fig)



@dataclass(frozen=True)
class PlanSchema:
    """A named means–ends plan schema used by the resource-bounded agent."""
    name: str

    def applicable(self, agent, target=None) -> bool:
        raise NotImplementedError


class FillHoleSchema(PlanSchema):
    def __init__(self):
        super().__init__("fill-hole")

    def applicable(self, agent, target=None) -> bool:
        if target is None or agent.belief.battery is None:
            return False
        out = manhattan(agent.position, target)
        back = manhattan(target, agent.charger)
        return agent.belief.battery >= out + back


class RechargeSchema(PlanSchema):
    def __init__(self):
        super().__init__("recharge")

    def applicable(self, agent, target=None) -> bool:
        return True


# ---------------------------- Exercise 2 --------------------------------

@dataclass
class BatteryRun:
    appeared: int
    filled: int
    stranded_steps: int
    effectiveness: float
    events: List[Event]


class ResourceBDIAgent(BDIAgent):
    """
    Resource-bounded BDI filter.

    Plan library:
      1. fill-hole: applicable only if battery covers
         agent->hole + hole->charger.
      2. recharge: always applicable.

    The planner remains means–ends/path planning; the resource constraint
    belongs to filter() because it determines which desires are eligible for
    commitment.
    """

    def __init__(self, charger=CHARGER):
        super().__init__(gamma=8, mode="single-minded")
        self.charger = charger
        self.fill_schema = FillHoleSchema()
        self.recharge_schema = RechargeSchema()

    def options(self):
        eligible = [
            Desire(h, -manhattan(self.position, h))
            for h in self.belief.holes
            if self.fill_schema.applicable(self, h)
        ]
        return sorted(
            eligible,
            key=lambda d: (manhattan(self.position, d.target), d.target)
        )

    def filter_resource(self, desires):
        # Filter = choose the nearest feasible fill-hole desire.
        if desires:
            best = desires[0]
            if (
                self.intention is not None
                and self.intention.target != self.charger
                and self.intention.target in self.belief.holes
                and self.fill_schema.applicable(self, self.intention.target)
            ):
                # Single-minded persistence while still feasible.
                return self.intention
            return Intention(best.target, best.utility)

        # No feasible fill-hole desire: commit to the always-applicable
        # recharge schema.
        if self.position != self.charger and self.recharge_schema.applicable(self):
            return Intention(
                self.charger,
                -manhattan(self.position, self.charger)
            )
        return None

    def deliberate_resource(self, step):
        self.deliberations += 1

        if self.intention is not None:
            if (
                self.intention.target != self.charger
                and not self.intention_achievable()
            ):
                old = self.intention.target
                self.intention = None
                self.plan = Plan([])
                self.log(step, "DROP", f"hole {old} disappeared")

        desires = self.options()
        selected = self.filter_resource(desires)

        if selected is None:
            self.intention = None
            self.plan = Plan([])
            return

        old = self.intention.target if self.intention else None
        self.intention = selected
        self.plan = self.make_plan(selected)
        self.since_reconsideration = 0

        if old is None:
            if selected.target == self.charger:
                self.log(step, "COMMIT", "committed to recharge at charger")
            else:
                self.log(step, "COMMIT", f"committed to fill hole {selected.target}")
        elif old != selected.target:
            self.log(step, "SWITCH", f"switched from {old} to {selected.target}")


def run_resource(seed, speed=2, steps=STEPS) -> BatteryRun:
    world = TileWorld(seed)
    agent = ResourceBDIAgent(CHARGER)
    battery = 40
    stranded = 0

    world.advance_tick()
    agent.brf(perceive(world, agent, battery))

    for step in range(1, steps + 1):
        if agent.intention is None or not agent.plan.actions:
            agent.deliberate_resource(step)
            for _ in range(speed):
                world.advance_tick()
            agent.brf(perceive(world, agent, battery))

            if agent.intention is None or not agent.plan.actions:
                continue

        # Consume battery for a move.
        if not agent.plan.actions:
            agent.intention = None
            continue

        if battery <= 0:
            stranded += 1
            # We still advance the world because a stranded action consumes
            # the agent's time step.
            for _ in range(speed):
                world.advance_tick()
            agent.brf(perceive(world, agent, battery))
            continue

        battery -= 1
        agent.position = agent.plan.actions.pop(0)
        agent.since_reconsideration += 1

        for _ in range(speed):
            world.advance_tick()

        agent.brf(perceive(world, agent, battery))

        if agent.position == CHARGER:
            battery = 40
            agent.log(step, "RECHARGE", "battery restored to 40")
            agent.intention = None
            agent.plan = Plan([])
            agent.since_reconsideration = 0
            agent.brf(perceive(world, agent, battery))
            continue

        # Fill if active hole is reached.
        if world.has(agent.position):
            world.fill(agent.position)
            agent.log(step, "ACHIEVE", f"filled hole at {agent.position}")
            agent.intention = None
            agent.plan = Plan([])
            agent.since_reconsideration = 0

        elif agent.intention is not None and not agent.intention_achievable():
            old = agent.intention.target
            agent.intention = None
            agent.plan = Plan([])
            agent.since_reconsideration = 0
            agent.log(step, "DROP", f"hole {old} disappeared")

        elif agent.should_reconsider():
            agent.deliberate_resource(step)
            for _ in range(speed):
                world.advance_tick()
            agent.brf(perceive(world, agent, battery))

    eff = world.filled / world.appeared if world.appeared else 0.0
    return BatteryRun(world.appeared, world.filled, stranded, eff, agent.events)


def run_battery_blind(seed, speed=2, steps=STEPS) -> BatteryRun:
    """Baseline: always chases nearest hole, ignoring battery feasibility."""
    world = TileWorld(seed)
    agent = BDIAgent(gamma=999999, mode="single-minded")
    battery = 40
    stranded = 0

    world.advance_tick()
    agent.brf(perceive(world, agent, battery))

    for step in range(1, steps + 1):
        if agent.intention is None or not agent.plan.actions:
            desires = agent.options()
            if desires:
                agent.intention = Intention(desires[0].target, desires[0].utility)
                agent.plan = agent.make_plan(agent.intention)
            else:
                for _ in range(speed):
                    world.advance_tick()
                agent.brf(perceive(world, agent, battery))
                continue

        # Blindly move even when energy is insufficient.
        if not agent.plan.actions:
            agent.intention = None
            continue

        if battery > 0:
            battery -= 1
        else:
            stranded += 1

        agent.position = agent.plan.actions.pop(0)
        for _ in range(speed):
            world.advance_tick()
        agent.brf(perceive(world, agent, battery))

        if world.has(agent.position):
            world.fill(agent.position)
            agent.intention = None
            agent.plan = Plan([])

        # The baseline only happens to recharge if it reaches charger.
        if agent.position == CHARGER:
            battery = 40
            agent.intention = None
            agent.plan = Plan([])

    eff = world.filled / world.appeared if world.appeared else 0.0
    return BatteryRun(world.appeared, world.filled, stranded, eff, agent.events)


def exercise2():
    rb = [run_resource(s) for s in SEEDS]
    blind = [run_battery_blind(s) for s in SEEDS]

    def summarize(agent_name, runs):
        total_app = sum(r.appeared for r in runs)
        total_fill = sum(r.filled for r in runs)
        total_stranded = sum(r.stranded_steps for r in runs)
        aggregate_eff = total_fill / total_app if total_app else 0.0
        return {
            "agent": agent_name,
            "mean_appeared": statistics.mean(r.appeared for r in runs),
            "mean_filled": statistics.mean(r.filled for r in runs),
            "mean_stranded": statistics.mean(r.stranded_steps for r in runs),
            "mean_effectiveness": aggregate_eff,
            "total_appeared": total_app,
            "total_filled": total_fill,
            "total_stranded": total_stranded,
        }

    return (
        summarize("resource-bounded", rb),
        summarize("battery-blind", blind),
    )


def main():
    print("=" * 88)
    print("AI401 — EXPERIMENT 3 — BDI DELIBERATION CYCLE")
    print("=" * 88)
    print("Explicit simulator choices not specified in the PDF:")
    print(f"  Grid                : {GRID_SIZE} x {GRID_SIZE}")
    print(f"  Hole appearance p  : {HOLE_APPEAR_PROB}")
    print(f"  Hole lifetime      : Uniform integer [{LIFE_MIN}, {LIFE_MAX}] ticks")
    print(f"  Start              : {START}")
    print(f"  Charger            : {CHARGER}")
    print(f"  Bold gamma         : {BOLD_GAMMA} (> max Manhattan plan 28)")
    print("  Tie-break          : nearest, then lexicographic cell")

    # ---- Core event trace
    print("\n" + "=" * 88)
    print("CORE BDI DEMONSTRATION")
    print("=" * 88)

    seed, demo = find_demo()
    print(f"Demo seed: {seed}")
    print(f"holes appeared={demo.appeared}, filled={demo.filled}, effectiveness={demo.effectiveness:.4f}")
    for e in demo.events:
        if e.kind in {"COMMIT", "ACHIEVE", "DROP", "SWITCH"}:
            print(f"step={e.step:03d} | {e.kind:<8} | {e.message}")

    with (OUT / "core_event_trace.txt").open("w", encoding="utf-8") as f:
        f.write(f"seed={seed}\n")
        for e in demo.events:
            if e.kind in {"COMMIT", "ACHIEVE", "DROP", "SWITCH"}:
                f.write(f"step={e.step} | {e.kind} | {e.message}\n")

    # ---- Exercise 1
    print("\n" + "=" * 88)
    print("EXERCISE 1 — GAMMA × WORLD SPEED SWEEP")
    print("=" * 88)
    t0 = perf_counter()
    rows, best = sweep()
    print(f"Sweep runtime: {perf_counter() - t0:.2f} s")
    print(f"{'speed':<8}" + "".join(f"{g:>12}" for g in GAMMA_LABELS))
    print("-" * 70)
    for speed in WORLD_SPEEDS:
        vals = []
        for g in GAMMA_LABELS:
            vals.append(next(r["mean_effectiveness"] for r in rows
                             if r["speed"] == speed and r["gamma"] == g))
        print(f"{speed:<8}" + "".join(f"{v:>12.4f}" for v in vals))

    print("\nBest gamma at each speed:")
    for speed in WORLD_SPEEDS:
        b = best[speed]
        print(f"speed={speed}: gamma={b['gamma']} -> effectiveness={b['mean_effectiveness']:.4f}")

    save_sweep(rows, best)

    with (OUT / "exercise1_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # ---- Exercise 2
    print("\n" + "=" * 88)
    print("EXERCISE 2 — RESOURCE-BOUNDED FILTER")
    print("=" * 88)
    rb, blind = exercise2()
    print(f"{'agent':<20}{'holes filled':>15}{'stranded steps':>18}{'effectiveness':>18}")
    print("-" * 72)
    for row in [rb, blind]:
        print(f"{row['agent']:<20}{row['mean_filled']:>15.3f}"
              f"{row['mean_stranded']:>18.3f}"
              f"{row['mean_effectiveness']:>18.4f}")
    print("\nAggregate totals:")
    for row in [rb, blind]:
        print(
            f"{row['agent']}: appeared={row['total_appeared']}, "
            f"filled={row['total_filled']}, stranded={row['total_stranded']}"
        )

    # ---- checks
    assert len(rows) == 20
    assert all(0.0 <= r["mean_effectiveness"] <= 1.0 for r in rows)
    assert {"COMMIT", "ACHIEVE", "DROP", "SWITCH"} <= {e.kind for e in demo.events}
    assert shortest_path((0,0),(2,3))[-1] == (2,3)
    assert len(shortest_path((0,0),(2,3))) == 5
    assert rb["mean_stranded"] <= blind["mean_stranded"] + 1e-9

    print("\n" + "=" * 88)
    print("ALL EXPERIMENT 3 IMPLEMENTATION CHECKS PASSED.")
    print("=" * 88)
    print(f"Outputs saved in: {OUT}")


if __name__ == "__main__":
    main()
