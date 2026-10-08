"""
AI401 – Intelligent Multi-Agent and Expert Systems
Experiment 1: Knowledge Representation — Semantic Network with Inheritance

Complete implementation:
1. Semantic network using (subject, relation, object) triples.
2. Default inheritance with exceptions.
3. Ostrich extension.
4. Frame-based representation using dictionaries.
5. Multiple inheritance with deterministic, justified parent-search order.

The implementation is intentionally self-contained and uses only the Python standard library.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


Triple = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# PART A — SEMANTIC NETWORK
# ---------------------------------------------------------------------------

class SemanticNetwork:
    """Semantic network represented as a list of (subject, relation, object) triples."""

    def __init__(self, triples: Optional[List[Triple]] = None) -> None:
        self.triples: List[Triple] = triples or []

    def add(self, subject: str, relation: str, obj: str) -> None:
        self.triples.append((subject, relation, obj))

    def parents(self, node: str) -> List[str]:
        """Return direct is-a parents of a node."""
        return [obj for subj, rel, obj in self.triples
                if subj == node and rel == "is-a"]

    def inheritance_chain(self, node: str) -> List[str]:
        """
        Follow a single-parent is-a chain upward.
        For the basic network, each node has at most one parent.
        """
        chain = [node]
        current = node
        visited = {node}

        while True:
            parent_list = self.parents(current)
            if not parent_list:
                break

            parent = parent_list[0]

            # Protect against accidental cycles in the knowledge base.
            if parent in visited:
                raise ValueError(f"Cycle detected in is-a hierarchy: {' -> '.join(chain + [parent])}")

            chain.append(parent)
            visited.add(parent)
            current = parent

        return chain

    def local_value(self, node: str, relation: str) -> Optional[str]:
        """Return the local value for a relation, if present."""
        for subj, rel, obj in self.triples:
            if subj == node and rel == relation:
                return obj
        return None

    def has_fact(self, node: str, relation: str, obj: Optional[str] = None) -> bool:
        """Check whether an exact triple exists."""
        return any(
            subj == node and rel == relation and (obj is None or value == obj)
            for subj, rel, value in self.triples
        )

    def query(self, node: str, relation: str, obj: Optional[str] = None) -> str:
        """
        Answer a query using nearest-first inheritance.

        Examples:
            query("penguin", "can")                  -> "cannot fly"
            query("ostrich", "can", "run_fast")      -> "can run_fast"

        The optional object removes ambiguity when a concept has more than
        one capability. Exceptions ("cannot") are always checked before
        normal values at every inheritance level.
        """
        chain = self.inheritance_chain(node)

        for current in chain:
            exception = self.local_value(current, "cannot")

            if exception is not None and relation == "can":
                if obj is None or obj == exception:
                    return f"cannot {exception}"

            value = self.local_value(current, relation)
            if value is not None and (obj is None or obj == value):
                return f"{relation} {value}"

        return "unknown"



def build_basic_network() -> SemanticNetwork:
    """
    Knowledge base suggested by the theory in the assignment:
        animal -> has skin
        fish   -> is-an animal
        shark  -> is-a fish
        bird   -> can fly
        penguin -> is-a bird
        penguin -> cannot fly
    """
    kb = SemanticNetwork()

    kb.add("animal", "has", "skin")
    kb.add("fish", "is-a", "animal")
    kb.add("shark", "is-a", "fish")

    kb.add("bird", "can", "fly")
    kb.add("penguin", "is-a", "bird")
    kb.add("penguin", "cannot", "fly")

    return kb


def print_network(kb: SemanticNetwork) -> None:
    print("KNOWLEDGE BASE (TRIPLES)")
    for triple in kb.triples:
        print(f"  {triple}")


# ---------------------------------------------------------------------------
# PART B — EXERCISE 1: OSTRICH
# ---------------------------------------------------------------------------

def extend_with_ostrich(kb: SemanticNetwork) -> None:
    """
    Ostrich:
      ostrich is-a bird
      ostrich cannot fly
      ostrich can run fast
    """
    kb.add("ostrich", "is-a", "bird")
    kb.add("ostrich", "cannot", "fly")
    kb.add("ostrich", "can", "run_fast")


# ---------------------------------------------------------------------------
# PART C — EXERCISE 2: FRAMES
# ---------------------------------------------------------------------------

class FrameSystem:
    """
    Frame representation.

    Example:
      frames["bird"] = {
          "ako": None,
          "can": "fly"
      }

      frames["penguin"] = {
          "ako": "bird",
          "cannot": "fly"
      }

    'ako' means 'a-kind-of' and behaves like an is-a link.
    """

    def __init__(self) -> None:
        self.frames: Dict[str, Dict[str, Optional[str]]] = {}

    def add_frame(self, concept: str, slots: Optional[Dict[str, Optional[str]]] = None) -> None:
        self.frames[concept] = slots.copy() if slots else {}

    def get_parent(self, concept: str) -> Optional[str]:
        frame = self.frames.get(concept, {})
        return frame.get("ako")

    def inheritance_chain(self, concept: str) -> List[str]:
        chain = [concept]
        current = concept
        visited = {concept}

        while True:
            parent = self.get_parent(current)
            if parent is None:
                break

            if parent in visited:
                raise ValueError(f"Cycle detected in frame hierarchy: {' -> '.join(chain + [parent])}")

            chain.append(parent)
            visited.add(parent)
            current = parent

        return chain

    def lookup(
        self,
        concept: str,
        slot: str,
        expected: Optional[str] = None
    ) -> str:
        """
        Frame inheritance lookup.

        expected can be used to disambiguate multiple fillers, e.g.
        lookup("ostrich", "can", "run_fast").
        """
        chain = self.inheritance_chain(concept)

        for current in chain:
            frame = self.frames.get(current, {})

            exception_value = frame.get("cannot")
            if slot == "can" and exception_value is not None:
                if expected is None or expected == exception_value:
                    return f"cannot {exception_value}"

            if slot in frame and frame[slot] is not None:
                value = frame[slot]
                if expected is None or expected == value:
                    return f"{slot} {value}"

        return "unknown"


    def print_frames(self) -> None:
        print("FRAMES")
        for concept, slots in self.frames.items():
            print(f"  {concept}: {slots}")


def build_frame_system() -> FrameSystem:
    fs = FrameSystem()

    fs.add_frame("animal", {"skin": "present"})
    fs.add_frame("fish", {"ako": "animal"})
    fs.add_frame("shark", {"ako": "fish"})

    fs.add_frame("bird", {"can": "fly"})
    fs.add_frame("penguin", {"ako": "bird", "cannot": "fly"})
    fs.add_frame("ostrich", {"ako": "bird", "cannot": "fly", "can": "run_fast"})

    return fs


# ---------------------------------------------------------------------------
# PART D — EXERCISE 3: MULTIPLE INHERITANCE
# ---------------------------------------------------------------------------

class MultiInheritanceNetwork(SemanticNetwork):
    """
    Semantic network allowing multiple parents.

    Search policy:
      Breadth-first search (BFS) from the most specific node.
      Direct parent order is the insertion order in the triple list.

    Why BFS?
      It searches all nodes at the same inheritance distance before moving
      farther away, so a property on a nearer parent can beat a property on
      a more distant ancestor. The order is deterministic because direct
      parents retain the order in which they were stored.
    """

    def inheritance_levels(self, node: str) -> List[List[str]]:
        levels: List[List[str]] = [[node]]
        visited = {node}
        frontier = [node]

        while frontier:
            next_frontier: List[str] = []

            for current in frontier:
                for parent in self.parents(current):
                    if parent not in visited:
                        visited.add(parent)
                        next_frontier.append(parent)

            if next_frontier:
                levels.append(next_frontier)

            frontier = next_frontier

        return levels

    def inheritance_chain(self, node: str) -> List[str]:
        """
        Return a flattened BFS traversal for display.
        """
        return [n for level in self.inheritance_levels(node) for n in level]

    def query(
        self,
        node: str,
        relation: str,
        obj: Optional[str] = None
    ) -> str:
        """
        Multiple-inheritance query using breadth-first search.

        At each BFS level:
          1. all exceptions are checked first;
          2. then all normal values are checked.

        This guarantees that a more specific exception/value is preferred
        over a more distant ancestor.
        """
        levels = self.inheritance_levels(node)

        for level in levels:
            for current in level:
                exception = self.local_value(current, "cannot")
                if relation == "can" and exception is not None:
                    if obj is None or obj == exception:
                        return f"cannot {exception}"

            for current in level:
                value = self.local_value(current, relation)
                if value is not None and (obj is None or obj == value):
                    return f"{relation} {value}"

        return "unknown"



def build_multiple_inheritance_network() -> MultiInheritanceNetwork:
    kb = MultiInheritanceNetwork()

    kb.add("animal", "has", "skin")
    kb.add("bird", "can", "fly")
    kb.add("fish", "is-a", "animal")
    kb.add("fish", "can", "swim")

    # Multiple parents:
    # flying_fish -> bird, fish
    kb.add("flying_fish", "is-a", "bird")
    kb.add("flying_fish", "is-a", "fish")

    # A more specific exception:
    kb.add("penguin", "is-a", "bird")
    kb.add("penguin", "cannot", "fly")

    return kb


# ---------------------------------------------------------------------------
# DEMONSTRATION / TESTS
# ---------------------------------------------------------------------------

def run_demo() -> None:
    print("=" * 72)
    print("AI401 — EXPERIMENT 1")
    print("Knowledge Representation — Semantic Network with Inheritance")
    print("=" * 72)

    # Core implementation
    kb = build_basic_network()
    print("\n--- BASIC SEMANTIC NETWORK ---")
    print_network(kb)

    print("\nInheritance chain for shark:")
    print("  " + " -> ".join(kb.inheritance_chain("shark")))

    print("\nInheritance chain for penguin:")
    print("  " + " -> ".join(kb.inheritance_chain("penguin")))

    queries = [
        ("shark", "has"),
        ("shark", "can"),
        ("penguin", "can"),
        ("bird", "can"),
        ("lion", "can"),
    ]

    print("\nQuery results:")
    for node, relation in queries:
        print(f"  query({node!r}, {relation!r}) = {kb.query(node, relation)!r}")

    # Exercise 1
    print("\n--- EXERCISE 1: OSTRICH ---")
    extend_with_ostrich(kb)
    print("Inheritance chain for ostrich:")
    print("  " + " -> ".join(kb.inheritance_chain("ostrich")))

    ostrich_queries = [
        ("ostrich", "can", None),
        ("ostrich", "can", "run_fast"),
        ("ostrich", "can", "fly"),
        ("ostrich", "has", None),
    ]

    for node, relation, obj in ostrich_queries:
        if obj is None:
            result = kb.query(node, relation)
            print(f"  query({node!r}, {relation!r}) = {result!r}")
        else:
            result = kb.query(node, relation, obj)
            print(f"  query({node!r}, {relation!r}, {obj!r}) = {result!r}")

    # Exercise 2
    print("\n--- EXERCISE 2: FRAMES ---")
    fs = build_frame_system()
    fs.print_frames()

    print("\nFrame inheritance chain for penguin:")
    print("  " + " -> ".join(fs.inheritance_chain("penguin")))

    frame_queries = [
        ("shark", "skin", None),
        ("penguin", "can", None),
        ("bird", "can", None),
        ("fish", "skin", None),
        ("ostrich", "can", "run_fast"),
        ("ostrich", "can", "fly"),
    ]

    print("\nFrame query results:")
    for node, slot, expected in frame_queries:
        if expected is None:
            result = fs.lookup(node, slot)
            print(f"  lookup({node!r}, {slot!r}) = {result!r}")
        else:
            result = fs.lookup(node, slot, expected)
            print(f"  lookup({node!r}, {slot!r}, {expected!r}) = {result!r}")

    # Exercise 3
    print("\n--- EXERCISE 3: MULTIPLE INHERITANCE ---")
    mikb = build_multiple_inheritance_network()

    print("Inheritance levels for flying_fish:")
    for depth, level in enumerate(mikb.inheritance_levels("flying_fish")):
        print(f"  Level {depth}: {level}")

    print("\nMultiple-inheritance query results:")
    multi_queries = [
        ("flying_fish", "can"),
        ("flying_fish", "has"),
        ("penguin", "can"),
    ]

    for node, relation in multi_queries:
        print(f"  query({node!r}, {relation!r}) = {mikb.query(node, relation)!r}")

    # Assertions: these are correctness checks for the practical.
    assert kb.query("shark", "has") == "has skin"
    assert kb.query("shark", "can") == "unknown"
    assert kb.query("penguin", "can") == "cannot fly"
    assert kb.query("bird", "can") == "can fly"
    assert kb.query("lion", "can") == "unknown"

    assert kb.query("ostrich", "can") == "cannot fly"
    assert kb.query("ostrich", "can", "run_fast") == "can run_fast"
    assert kb.query("ostrich", "can", "fly") == "cannot fly"
    assert kb.query("ostrich", "has") == "unknown"

    assert fs.lookup("shark", "skin") == "skin present"
    assert fs.lookup("penguin", "can") == "cannot fly"
    assert fs.lookup("bird", "can") == "can fly"
    assert fs.lookup("fish", "skin") == "skin present"
    assert fs.lookup("ostrich", "can", "run_fast") == "can run_fast"
    assert fs.lookup("ostrich", "can", "fly") == "cannot fly"

    assert mikb.query("flying_fish", "can") == "can fly"
    assert mikb.query("flying_fish", "has") == "has skin"
    assert mikb.query("penguin", "can") == "cannot fly"

    print("\nALL CORRECTNESS TESTS PASSED.")


if __name__ == "__main__":
    run_demo()