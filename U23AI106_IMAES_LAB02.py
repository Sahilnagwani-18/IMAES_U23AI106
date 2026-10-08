"""
AI401 – Intelligent Multi-Agent and Expert Systems
Experiment 2: Production Rules and the Match Phase

Complete implementation of:
1. IF–THEN production rules.
2. Working memory (WM) as a set of facts.
3. Match phase / conflict-set computation.
4. Refractoriness: a rule is not applicable when its conclusion is already in WM.
5. Exercise 1: add facts one at a time and record conflict-set changes.
6. Exercise 2: negated conditions of the form "NOT fact".
7. Exercise 3: benchmark match time with 1,000 auto-generated rules and explain
   the motivation for the RETE algorithm.

Knowledge base choice:
A smart-home security/automation domain was selected because it naturally
contains positive and negative conditions and gives useful demonstrations of
the match phase. The matcher itself is domain-independent.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import FrozenSet, Iterable, List, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# DATA MODEL
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """
    A rule condition.

    positive=True:
        fact must be present in working memory.

    positive=False:
        fact must NOT be present in working memory.
    """
    fact: str
    positive: bool = True

    def __str__(self) -> str:
        return self.fact if self.positive else f"NOT {self.fact}"


@dataclass(frozen=True)
class Rule:
    """An IF–THEN production rule."""
    rule_id: str
    conditions: Tuple[Condition, ...]
    conclusion: str

    def __str__(self) -> str:
        lhs = " AND ".join(str(c) for c in self.conditions)
        return f"{self.rule_id}: IF {lhs} THEN {self.conclusion}"


# ---------------------------------------------------------------------------
# PRODUCTION SYSTEM / MATCHER
# ---------------------------------------------------------------------------

class ProductionMatcher:
    """
    Implements the match phase of a production system.

    A rule is in the conflict set exactly when:
      1. every positive condition is in WM;
      2. every negated condition is absent from WM;
      3. the conclusion is NOT already in WM (refractoriness).
    """

    def __init__(self, rules: Iterable[Rule], working_memory: Iterable[str] = ()) -> None:
        self.rules: List[Rule] = list(rules)
        self.wm: Set[str] = set(working_memory)

    def set_working_memory(self, facts: Iterable[str]) -> None:
        self.wm = set(facts)

    def add_fact(self, fact: str) -> bool:
        """Add one fact. Return True iff WM changed."""
        old_size = len(self.wm)
        self.wm.add(fact)
        return len(self.wm) != old_size

    def remove_fact(self, fact: str) -> bool:
        """Remove one fact. Return True iff WM changed."""
        if fact in self.wm:
            self.wm.remove(fact)
            return True
        return False

    def condition_matches(self, condition: Condition) -> bool:
        """Check one condition against current WM."""
        present = condition.fact in self.wm
        return present if condition.positive else not present

    def rule_matches(self, rule: Rule) -> bool:
        """
        Return True if the rule belongs to the conflict set.

        All conditions are conjunctive (AND).
        Refractoriness is checked after the conditions.
        """
        if rule.conclusion in self.wm:
            return False

        return all(self.condition_matches(condition)
                   for condition in rule.conditions)

    def conflict_set(self) -> List[Rule]:
        """Return all currently applicable rules, preserving rule order."""
        return [rule for rule in self.rules if self.rule_matches(rule)]

    def conflict_set_ids(self) -> List[str]:
        return [rule.rule_id for rule in self.conflict_set()]

    def explain_rule(self, rule: Rule) -> str:
        """Explain why a rule matches or does not match."""
        missing_positive = [
            condition.fact
            for condition in rule.conditions
            if condition.positive and condition.fact not in self.wm
        ]
        violated_negative = [
            condition.fact
            for condition in rule.conditions
            if not condition.positive and condition.fact in self.wm
        ]

        if rule.conclusion in self.wm:
            return f"{rule.rule_id}: NOT MATCHED — conclusion already in WM (refractory)."
        if missing_positive:
            return (
                f"{rule.rule_id}: NOT MATCHED — missing positive condition(s): "
                + ", ".join(missing_positive)
            )
        if violated_negative:
            return (
                f"{rule.rule_id}: NOT MATCHED — negated condition(s) violated: "
                + ", ".join(violated_negative)
            )
        return f"{rule.rule_id}: MATCHED — all conditions satisfied."


# ---------------------------------------------------------------------------
# KNOWLEDGE BASE
# ---------------------------------------------------------------------------

def P(fact: str) -> Condition:
    return Condition(fact, positive=True)


def N(fact: str) -> Condition:
    return Condition(fact, positive=False)


def build_smart_home_rules() -> List[Rule]:
    """
    Domain knowledge: smart-home security and automation.

    The rules intentionally create:
      - individual matches,
      - multi-condition matches,
      - negated-condition matches,
      - a rule blocked by refractoriness.
    """
    return [
        Rule("R1",  (P("person_detected"),), "camera_alert"),
        Rule("R2",  (P("person_detected"), P("home_armed")), "security_alert"),
        Rule("R3",  (P("door_open"), P("home_armed")), "door_alarm"),
        Rule("R4",  (P("motion_detected"), P("night_mode")), "lights_on"),
        Rule("R5",  (P("smoke_detected"),), "fire_alarm"),
        Rule("R6",  (P("water_leak"), N("maintenance_mode")), "water_alarm"),
        Rule("R7",  (P("package_detected"), N("package_collected")), "delivery_alert"),
        Rule("R8",  (P("window_open"), N("home_occupied")), "window_alert"),
        Rule("R9",  (P("security_alert"),), "notify_owner"),
        Rule("R10", (P("camera_alert"), N("privacy_mode")), "record_video"),
        Rule("R11", (P("home_armed"), N("maintenance_mode")), "armed_monitoring"),
        Rule("R12", (P("battery_low"),), "low_battery_alert"),
    ]


# ---------------------------------------------------------------------------
# EXERCISE 1
# Add facts one at a time and record the conflict-set evolution.
# ---------------------------------------------------------------------------

def conflict_set_growth_demo() -> List[Tuple[int, str, List[str]]]:
    rules = build_smart_home_rules()
    matcher = ProductionMatcher(rules)

    sequence = [
        "home_armed",
        "person_detected",
        "door_open",
        "motion_detected",
        "night_mode",
        "maintenance_mode",
        "water_leak",
        "package_detected",
        "package_collected",
        "privacy_mode",
        "smoke_detected",
        "battery_low",
    ]

    history: List[Tuple[int, str, List[str]]] = []

    print("\n" + "=" * 78)
    print("EXERCISE 1 — WORKING MEMORY UPDATED ONE FACT AT A TIME")
    print("=" * 78)

    print(f"{'Step':<5} {'Added fact':<22} Conflict set")
    print("-" * 78)

    for step, fact in enumerate(sequence, start=1):
        matcher.add_fact(fact)
        conflicts = matcher.conflict_set_ids()
        history.append((step, fact, conflicts.copy()))
        print(f"{step:<5} {fact:<22} {conflicts}")

    print("\nObservation:")
    print("The conflict set changes after each WM update. Adding a required")
    print("positive fact can activate a rule; adding a fact that violates a")
    print("NOT-condition can deactivate a rule. Adding a conclusion already")
    print("required by a matching rule demonstrates refractoriness.")

    return history


# ---------------------------------------------------------------------------
# EXERCISE 2
# Negated conditions: NOT fact
# ---------------------------------------------------------------------------

def negation_demo() -> None:
    rules = [
        Rule("N1", (P("water_leak"), N("maintenance_mode")), "water_alarm"),
        Rule("N2", (P("package_detected"), N("package_collected")), "delivery_alert"),
        Rule("N3", (P("camera_alert"), N("privacy_mode")), "record_video"),
    ]

    matcher = ProductionMatcher(rules, {"water_leak", "package_detected", "camera_alert"})

    print("\n" + "=" * 78)
    print("EXERCISE 2 — NEGATED CONDITIONS")
    print("=" * 78)

    print("Initial WM:", sorted(matcher.wm))
    print("Initial conflict set:", matcher.conflict_set_ids())

    matcher.add_fact("maintenance_mode")
    print("\nAfter adding 'maintenance_mode':")
    print("Conflict set:", matcher.conflict_set_ids())
    print("N1 explanation:", matcher.explain_rule(rules[0]))

    matcher.add_fact("package_collected")
    print("\nAfter adding 'package_collected':")
    print("Conflict set:", matcher.conflict_set_ids())
    print("N2 explanation:", matcher.explain_rule(rules[1]))

    matcher.add_fact("privacy_mode")
    print("\nAfter adding 'privacy_mode':")
    print("Conflict set:", matcher.conflict_set_ids())
    print("N3 explanation:", matcher.explain_rule(rules[2]))

    assert matcher.conflict_set_ids() == []


# ---------------------------------------------------------------------------
# REFRACTORINESS DEMO
# ---------------------------------------------------------------------------

def refractoriness_demo() -> None:
    print("\n" + "=" * 78)
    print("REFRACTORINESS DEMONSTRATION")
    print("=" * 78)

    rule = Rule("RF1", (P("person_detected"),), "camera_alert")
    matcher = ProductionMatcher([rule], {"person_detected"})

    print("WM before conclusion is added:", sorted(matcher.wm))
    print("Conflict set:", matcher.conflict_set_ids())

    matcher.add_fact("camera_alert")

    print("\nWM after adding the conclusion 'camera_alert':", sorted(matcher.wm))
    print("Conflict set:", matcher.conflict_set_ids())
    print(matcher.explain_rule(rule))

    assert matcher.conflict_set_ids() == []


# ---------------------------------------------------------------------------
# EXERCISE 3
# Benchmark with 1,000 auto-generated rules using time.perf_counter.
# ---------------------------------------------------------------------------

def generate_benchmark_rules(num_rules: int) -> List[Rule]:
    """Generate deterministic rules for the performance experiment."""
    rules: List[Rule] = []

    for i in range(num_rules):
        # Two positive conditions give a representative non-trivial matcher.
        # All rules below will match, so the benchmark measures full scanning.
        rules.append(
            Rule(
                f"B{i:04d}",
                (P(f"sensor_{i}"), P("system_ready")),
                f"action_{i}",
            )
        )

    return rules


def benchmark(num_rules: int, repetitions: int = 100) -> float:
    """
    Return average conflict-set matching time in milliseconds.
    """
    rules = generate_benchmark_rules(num_rules)

    # All sensor_i facts and system_ready are present, so every rule matches.
    wm = {f"sensor_{i}" for i in range(num_rules)}
    wm.add("system_ready")

    matcher = ProductionMatcher(rules, wm)

    # Warm-up.
    matcher.conflict_set()

    start = perf_counter()
    last_count = 0

    for _ in range(repetitions):
        conflicts = matcher.conflict_set()
        last_count = len(conflicts)

    elapsed = perf_counter() - start
    average_ms = (elapsed / repetitions) * 1000.0

    assert last_count == num_rules
    return average_ms


def benchmark_experiment() -> List[Tuple[int, float]]:
    print("\n" + "=" * 78)
    print("EXERCISE 3 — MATCH-TIME SCALING")
    print("=" * 78)

    sizes = [100, 250, 500, 1000]
    results: List[Tuple[int, float]] = []

    print(f"{'Rules':<10} {'Average match time (ms)':>26}")
    print("-" * 40)

    for size in sizes:
        avg_ms = benchmark(size, repetitions=100)
        results.append((size, avg_ms))
        print(f"{size:<10} {avg_ms:>26.6f}")

    print("\nInterpretation:")
    print("A naive production-system matcher scans the rule base and tests")
    print("conditions for every cycle. Therefore, as the number of rules grows,")
    print("the amount of work grows approximately with the number of rules")
    print("(and with the number of conditions per rule).")
    print("\nWhy RETE exists:")
    print("RETE avoids recomputing all rule-condition matches from scratch after")
    print("every WM change. It shares common condition tests and maintains")
    print("partial matches in an incremental network, substantially reducing")
    print("repeated work in large production systems.")

    return results


# ---------------------------------------------------------------------------
# CORRECTNESS TESTS
# ---------------------------------------------------------------------------

def run_correctness_tests() -> None:
    rules = build_smart_home_rules()

    # 1. Basic positive matching.
    matcher = ProductionMatcher(rules, {"person_detected"})
    assert "R1" in matcher.conflict_set_ids()
    assert "R2" not in matcher.conflict_set_ids()

    # 2. Conjunctive matching.
    matcher.add_fact("home_armed")
    assert "R2" in matcher.conflict_set_ids()

    # 3. Negative condition.
    matcher = ProductionMatcher(
        rules,
        {"water_leak", "package_detected", "camera_alert"}
    )
    assert "R6" in matcher.conflict_set_ids()
    assert "R7" in matcher.conflict_set_ids()
    assert "R10" in matcher.conflict_set_ids()

    # 4. Negation should be broken by adding the forbidden fact.
    matcher.add_fact("maintenance_mode")
    matcher.add_fact("package_collected")
    matcher.add_fact("privacy_mode")
    assert "R6" not in matcher.conflict_set_ids()
    assert "R7" not in matcher.conflict_set_ids()
    assert "R10" not in matcher.conflict_set_ids()

    # 5. Refractoriness.
    matcher = ProductionMatcher(
        [Rule("X1", (P("a"),), "b")],
        {"a"}
    )
    assert matcher.conflict_set_ids() == ["X1"]
    matcher.add_fact("b")
    assert matcher.conflict_set_ids() == []

    # 6. Unknown / unsupported facts do not cause false positives.
    matcher = ProductionMatcher(rules, {"unrelated_fact"})
    assert matcher.conflict_set_ids() == []

    # 7. Benchmark correctness for 1000 rules.
    rules_1000 = generate_benchmark_rules(1000)
    wm_1000 = {f"sensor_{i}" for i in range(1000)} | {"system_ready"}
    matcher = ProductionMatcher(rules_1000, wm_1000)
    assert len(matcher.conflict_set()) == 1000

    print("\n" + "=" * 78)
    print("CORRECTNESS TESTS")
    print("=" * 78)
    print("All correctness tests passed.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("AI401 — EXPERIMENT 2")
    print("Production Rules and the Match Phase")
    print("=" * 78)

    rules = build_smart_home_rules()

    print("\n" + "=" * 78)
    print("KNOWLEDGE BASE — IF–THEN PRODUCTION RULES")
    print("=" * 78)
    for rule in rules:
        print(rule)

    print("\nInitial working memory: {'home_armed', 'person_detected'}")
    matcher = ProductionMatcher(rules, {"home_armed", "person_detected"})
    print("Conflict set:", matcher.conflict_set_ids())

    print("\nDetailed match explanations:")
    for rule in rules:
        print(matcher.explain_rule(rule))

    conflict_set_growth_demo()
    negation_demo()
    refractoriness_demo()
    benchmark_experiment()
    run_correctness_tests()

    print("\n" + "=" * 78)
    print("FINAL RESULT: EXPERIMENT 2 COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()