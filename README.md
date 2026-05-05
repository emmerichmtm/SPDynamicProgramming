Ultrametric Solow--Polasky Dynamic Programming
This directory contains a compact reference implementation of the exact dynamic
programming algorithm for fixed-cardinality Solow--Polasky diversity subset
selection on finite rooted ultrametric spaces.
The implementation accompanies the ultrametric section of the paper
> **Hard and Tractable Cases of Solow--Polasky Diversity Subset Selection in Finite Metric Spaces**
and is intended for small-scale reproduction, testing, and illustration of the
recurrence used in the proof.
File
```text
ultrametric_sp_dp_updated.py
```
The script is self-contained and uses only the Python standard library.
Problem
Given a finite ultrametric space represented by a rooted tree, a cardinality
parameter `k`, and a similarity parameter `theta > 0`, the task is to select
`k` leaves maximizing the Solow--Polasky diversity
```text
SP_theta(S) = 1^T Z(S)^{-1} 1,
```
where
```text
Z(S)_{xy} = exp(-theta * d(x,y)).
```
For a rooted ultrametric tree, the distance between two leaves is twice the
height of their least common ancestor.
Algorithmic idea
For an internal node `u` with children `u_1, ..., u_m`, let
```text
c_u = exp(-theta * 2 * h(u)).
```
If `a_i` is the optimal Solow--Polasky value obtained by selecting `r_i` leaves
inside child `u_i`, then the value for the combined selection is
```text
B / (1 + c_u B),
where B = sum_i a_i / (1 - c_u a_i).
```
The dynamic program maximizes the additive quantity `B` for every feasible
cardinality. It supports non-binary branching and stores one optimal subset for
reconstruction.
Requirements
Python 3.9 or newer is recommended.
No external packages are required.
Running the self-test
From this directory, run:
```bash
python ultrametric_sp_dp_updated.py
```
The script executes:
a four-leaf worked example;
deterministic random ultrametric instances;
brute-force validation by direct matrix inversion on small instances.
A typical output is:
```text
four-leaf example, k=2: DP=..., subset=(...); brute=...
four-leaf example, k=3: DP=..., subset=(...); brute=...
validated 48 random instances; max abs diff = ...
```
The final line reports the largest absolute difference between the dynamic
programming value and the brute-force value. Differences should be at the level
of floating-point roundoff.
Minimal usage example
```python
from ultrametric_sp_dp_updated import leaf, node, ultrametric_sp_dp

# Tree: ((0,1) at height 1.0, (2,3) at height 1.5) at height 2.5
left = node("u01", 1.0, leaf("0"), leaf("1"))
right = node("u23", 1.5, leaf("2"), leaf("3"))
root = node("root", 2.5, left, right)

best, tables = ultrametric_sp_dp(root, k=2, theta=0.7)

print(best.value)
print(best.subset)
```
Tree conventions
Leaves are created with `leaf(name)` and have height `0`.
Internal nodes are created with `node(name, height, *children)`.
Internal node heights must be strictly larger than the heights of all their
children.
Leaf labels must be unique.
Distances are defined as
```text
d(x,y) = 2 * height(lca(x,y))
```
for distinct leaves `x` and `y`, and `d(x,x)=0`.
Main functions
`ultrametric_sp_dp(root, k, theta)`
Computes an optimal fixed-cardinality subset.
Returns:
```python
best, tables
```
where `best` is a `DPCell` with fields
```python
best.value   # optimal Solow--Polasky diversity value
best.subset  # one optimal selected subset of leaf labels
```
and `tables` maps node names to dynamic-programming tables.
`sp_value_by_inversion(root, subset, theta)`
Evaluates the Solow--Polasky diversity of a given subset by direct linear solve.
This is used for validation.
`brute_force_best(root, k, theta)`
Enumerates all `k`-subsets and returns the best one. This is intended only for
small instances and reproducibility checks.
Reproducibility role
The implementation is not optimized for large-scale computation. Its purpose is
to make the ultrametric recurrence transparent and reproducible:
it follows the proof notation closely;
it validates the dynamic program against direct matrix inversion;
it provides deterministic random checks for small instances;
it reconstructs an optimal subset, not only the optimal value.
Citation note
When using this code in connection with the article, please cite the paper draft
or the corresponding submitted/published version once available.
