"""Exact dynamic programming for fixed-cardinality Solow--Polasky diversity
on finite rooted ultrametric spaces.

This is a compact reference implementation matching the ultrametric recurrence
used in the paper draft.

The model is a rooted tree whose leaves are the points.  Each internal node u
has a positive height h(u), strictly larger than the heights of its children.
The distance between two leaves is twice the height of their least common
ancestor.  For theta > 0 the similarity is exp(-theta * distance).

For an internal node u with children u_1,...,u_m and c_u = exp(-theta*2*h(u)),
if a_i is the optimal SP value obtained by selecting r_i leaves in child i,
then the parent value for the composition (r_1,...,r_m) is

    B / (1 + c_u B),       B = sum_i a_i / (1 - c_u a_i).

The dynamic program maximizes the additive quantity B for every cardinality.
It also stores one optimal selected subset for reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, isclose
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class UNode:
    """Node of a rooted ultrametric tree.

    Leaves have no children and should have height 0.  Internal nodes have
    strictly positive height and one or more children.  Binary and non-binary
    branching are both supported.
    """

    name: str
    height: float = 0.0
    children: Tuple["UNode", ...] = ()

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


def leaf(name: str) -> UNode:
    return UNode(name=name, height=0.0, children=())


def node(name: str, height: float, *children: UNode) -> UNode:
    if not children:
        raise ValueError("use leaf(name) for leaves")
    return UNode(name=name, height=float(height), children=tuple(children))


@dataclass(frozen=True)
class DPCell:
    value: float
    subset: Tuple[str, ...]


def leaf_labels(root: UNode) -> Tuple[str, ...]:
    labels: List[str] = []

    def visit(u: UNode) -> None:
        if u.is_leaf:
            labels.append(u.name)
        else:
            for v in u.children:
                visit(v)

    visit(root)
    if len(labels) != len(set(labels)):
        raise ValueError("leaf labels must be unique")
    return tuple(labels)


def validate_ultrametric_tree(root: UNode) -> None:
    """Check the height convention used by the DP."""

    def visit(u: UNode) -> float:
        if u.is_leaf:
            if u.height != 0:
                raise ValueError(f"leaf {u.name!r} has height {u.height}, expected 0")
            return 0.0
        if u.height <= 0:
            raise ValueError(f"internal node {u.name!r} must have positive height")
        max_child_height = max(visit(v) for v in u.children)
        if not (u.height > max_child_height):
            raise ValueError(
                f"node {u.name!r} has height {u.height}, not larger than child height {max_child_height}"
            )
        return u.height

    visit(root)


def ultrametric_sp_dp(root: UNode, k: int, theta: float) -> Tuple[DPCell, Dict[str, List[Optional[DPCell]]]]:
    """Return an optimal k-subset and all DP tables.

    Parameters
    ----------
    root:
        Root of the ultrametric tree.
    k:
        Required subset cardinality.
    theta:
        Positive Solow--Polasky similarity parameter.

    Returns
    -------
    best:
        A DPCell containing the optimal SP value and one optimal subset.
    tables:
        Dictionary mapping node names to their cardinality tables.  Entry t is
        either None, if t is infeasible below that node, or a DPCell.
    """

    if k < 0:
        raise ValueError("k must be nonnegative")
    if theta <= 0:
        raise ValueError("theta must be positive")
    validate_ultrametric_tree(root)
    n = len(leaf_labels(root))
    if k > n:
        raise ValueError(f"cannot choose k={k} leaves from n={n} leaves")

    tables: Dict[str, List[Optional[DPCell]]] = {}

    def solve(u: UNode) -> List[Optional[DPCell]]:
        if u.is_leaf:
            table: List[Optional[DPCell]] = [DPCell(0.0, tuple())]
            if k >= 1:
                table.append(DPCell(1.0, (u.name,)))
            tables[u.name] = table
            return table

        child_tables = [solve(v) for v in u.children]
        child_sizes = [len(tab) - 1 for tab in child_tables]
        max_size = min(k, sum(child_sizes))
        c = exp(-theta * 2.0 * u.height)

        # Knapsack over children, maximizing the additive B-score.  The stored
        # subset is used only for reconstruction and reproducibility checks.
        best_B = [float("-inf")] * (max_size + 1)
        best_subset: List[Optional[Tuple[str, ...]]] = [None] * (max_size + 1)
        best_B[0] = 0.0
        best_subset[0] = tuple()

        used_so_far = 0
        for tab in child_tables:
            new_B = [float("-inf")] * (max_size + 1)
            new_subset: List[Optional[Tuple[str, ...]]] = [None] * (max_size + 1)
            child_max = len(tab) - 1
            for used in range(min(used_so_far, max_size) + 1):
                if best_subset[used] is None:
                    continue
                for r in range(min(child_max, max_size - used) + 1):
                    cell = tab[r]
                    if cell is None:
                        continue
                    a = cell.value
                    if r == 0:
                        contribution = 0.0
                    else:
                        denom = 1.0 - c * a
                        # The separation lemma guarantees positivity in exact
                        # arithmetic.  This guard catches malformed trees or
                        # severe floating-point problems.
                        if denom <= 0.0:
                            raise ArithmeticError(
                                f"nonpositive denominator at node {u.name!r}: 1 - c*a = {denom}"
                            )
                        contribution = a / denom
                    total_r = used + r
                    candidate_B = best_B[used] + contribution
                    if candidate_B > new_B[total_r]:
                        new_B[total_r] = candidate_B
                        new_subset[total_r] = tuple(sorted(best_subset[used] + cell.subset))
            used_so_far = min(max_size, used_so_far + child_max)
            best_B, best_subset = new_B, new_subset

        table = [None] * (max_size + 1)
        for r in range(max_size + 1):
            if best_subset[r] is None:
                continue
            B = best_B[r]
            value = 0.0 if r == 0 else B / (1.0 + c * B)
            table[r] = DPCell(value=value, subset=best_subset[r])
        tables[u.name] = table
        return table

    root_table = solve(root)
    best = root_table[k]
    if best is None:
        raise RuntimeError("internal error: requested cardinality is infeasible")
    return best, tables


# --- Direct Solow--Polasky evaluation and brute-force checks -----------------


def _leaf_paths(root: UNode) -> Dict[str, Tuple[UNode, ...]]:
    paths: Dict[str, Tuple[UNode, ...]] = {}

    def visit(u: UNode, path: Tuple[UNode, ...]) -> None:
        new_path = path + (u,)
        if u.is_leaf:
            paths[u.name] = new_path
        else:
            for v in u.children:
                visit(v, new_path)

    visit(root, tuple())
    return paths


def distance(root: UNode, x: str, y: str) -> float:
    if x == y:
        return 0.0
    paths = _leaf_paths(root)
    px, py = paths[x], paths[y]
    lca = px[0]
    for a, b in zip(px, py):
        if a == b:
            lca = a
        else:
            break
    return 2.0 * lca.height


def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting; sufficient for small checks."""
    n = len(A)
    M = [row[:] + [rhs] for row, rhs in zip(A, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(M[i][col]))
        if abs(M[pivot][col]) < 1e-14:
            raise ArithmeticError("singular or ill-conditioned matrix in brute-force check")
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
        pivot_value = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot_value
        for i in range(n):
            if i == col:
                continue
            factor = M[i][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                M[i][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def sp_value_by_inversion(root: UNode, subset: Sequence[str], theta: float) -> float:
    """Evaluate SP_theta(S) by direct matrix inversion/linear solve."""
    labels = tuple(subset)
    if len(labels) == 0:
        return 0.0
    m = len(labels)
    Z = [
        [exp(-theta * distance(root, labels[i], labels[j])) for j in range(m)]
        for i in range(m)
    ]
    x = _solve_linear_system(Z, [1.0] * m)
    return sum(x)


def brute_force_best(root: UNode, k: int, theta: float) -> DPCell:
    labels = leaf_labels(root)
    best_value = float("-inf")
    best_subset: Tuple[str, ...] = tuple()
    for subset in combinations(labels, k):
        value = sp_value_by_inversion(root, subset, theta)
        if value > best_value:
            best_value = value
            best_subset = tuple(subset)
    return DPCell(best_value, best_subset)


# --- Examples and deterministic validation -----------------------------------


def four_leaf_example() -> UNode:
    """The example ((0,1)@1.0,(2,3)@1.5)@2.5."""
    left = node("u01", 1.0, leaf("0"), leaf("1"))
    right = node("u23", 1.5, leaf("2"), leaf("3"))
    return node("root", 2.5, left, right)


def random_binary_ultrametric(n: int, seed: int) -> UNode:
    """Generate a deterministic random binary ultrametric tree with n leaves."""
    if n < 1:
        raise ValueError("n must be positive")
    rng = Random(seed)
    forest: List[UNode] = [leaf(str(i)) for i in range(n)]
    height = 0.0
    merge_index = 0
    while len(forest) > 1:
        i, j = sorted(rng.sample(range(len(forest)), 2), reverse=True)
        a = forest.pop(i)
        b = forest.pop(j)
        height += rng.uniform(0.2, 1.0)
        forest.append(node(f"m{merge_index}", height, a, b))
        merge_index += 1
    return forest[0]


def run_self_test() -> None:
    theta = 0.7
    root = four_leaf_example()
    for k in (2, 3):
        dp, _ = ultrametric_sp_dp(root, k, theta)
        brute = brute_force_best(root, k, theta)
        print(f"four-leaf example, k={k}: DP={dp.value:.12f}, subset={dp.subset}; brute={brute.value:.12f}")
        assert isclose(dp.value, brute.value, rel_tol=1e-12, abs_tol=1e-12)

    seeds = [7, 11, 17, 23, 31, 37, 41, 43]
    theta_values = [0.5, 0.8, 1.2]
    checked = 0
    max_abs_diff = 0.0
    for seed in seeds:
        for n in range(5, 11):
            theta = theta_values[(seed + n) % len(theta_values)]
            k = max(2, n // 2)
            root = random_binary_ultrametric(n, seed=1000 * seed + n)
            dp, _ = ultrametric_sp_dp(root, k, theta)
            brute = brute_force_best(root, k, theta)
            diff = abs(dp.value - brute.value)
            max_abs_diff = max(max_abs_diff, diff)
            if diff > 1e-10:
                raise AssertionError(
                    f"DP/brute-force mismatch for seed={seed}, n={n}, k={k}, theta={theta}: "
                    f"DP={dp.value}, brute={brute.value}"
                )
            checked += 1
    print(f"validated {checked} random instances; max abs diff = {max_abs_diff:.3e}")


if __name__ == "__main__":
    run_self_test()
