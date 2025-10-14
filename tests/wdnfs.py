from itertools import combinations

from direct_computation import Cube, WDNF

# My cubes
SIMPLE_WDNF = [(3, Cube([], [1])), (5, Cube([1], [3])), (3, Cube([2,3], [1]))]
WDNF_WITH_CUBE_WITH_SAME_LITERAL_BE_BOTH_POSITIVE_AND_NEGATIVE = SIMPLE_WDNF + [(6,Cube([1], [1]))]
CONSTANT = [(10, Cube([], []))]

MY_WDNFs = [WDNF(wdnf) for wdnf in [
    SIMPLE_WDNF, WDNF_WITH_CUBE_WITH_SAME_LITERAL_BE_BOTH_POSITIVE_AND_NEGATIVE, CONSTANT
]]

########################################################################################
# ChatGPT simple WDNFs
########################################################################################

# 1) Empty WDNF (always 0): no cubes at all.
EMPTY = []

# 2) Single-literal positive cube: fires iff x1=True.
SINGLE_POS = [(1.0, Cube([1], []))]

# 3) Single-literal negative cube: fires iff x2=False.
SINGLE_NEG = [(1.0, Cube([], [2]))]

# 4) Overlapping / subsuming cubes:
#    - First cube fires on x1
#    - Second is stricter (x1 & x2)
#    - Third requires x1 & ~x3 (overlap/conflict checks)
OVERLAPPING = [
    (2.0, Cube([1], [])),
    (3.0, Cube([1, 2], [])),
    (5.0, Cube([1], [3])),
]

# 5) Contradictory (unsatisfiable) cube alone: never fires.
UNSAT_CUBE_ONLY = [(4.0, Cube([1], [1]))]

# 6) Mix of satisfiable + unsatisfiable cubes:
#    - Second cube can never contribute; tests that it's effectively ignored at assign time.
MIXED_SAT_UNSAT = [
    (3.0, Cube([2], [])),
    (6.0, Cube([3], [3])),
]

# 7) Zero-weight cube: should have no effect on assign/calc_metric.
ZERO_WEIGHT = [
    (0.0, Cube([1, 2], [])),
    (5.0, Cube([], [3])),
]

# 8) Negative weight cube: tests aggregation with negative contributions.
NEGATIVE_WEIGHT = [
    (-2.5, Cube([2], [])),
    (1.0, Cube([2, 3], [])),
]

# 9) Duplicate literals in a cube: behavior should be identical to deduped sets.
DUP_LITERALS = [
    (1.0, Cube([1, 1, 2], [3, 3])),
]

# 10) Large/rare variable indices: tests variables() set-building and no assumptions on small indices.
SPARSE_INDICES = [
    (1.0, Cube([1000], [])),
    (2.0, Cube([], [999])),
]

# 11) Redundant duplicate cubes: same condition twice with different weights; tests numeric aggregation.
DUPLICATE_CUBES = [
    (2.0, Cube([1, 2], [])),
    (3.0, Cube([1, 2], [])),  # identical to above
]

# 12) Mutually exclusive cubes: at most one can fire at a time.
MUTUALLY_EXCLUSIVE = [
    (4.0, Cube([1], [2])),
    (7.0, Cube([2], [1])),
]

# 13) Tautology plus others: constant term + specific cube; tests constant shifting.
CONST_PLUS_SPECIFIC = [
    (10.0, Cube([], [])),        # tautology
    (3.0,  Cube([1], [3])),
]

# 14) All-negative literals cube: fires only when all listed vars are False.
ALL_NEG = [
    (5.0, Cube([], [1, 2, 3])),
]

# 15) Mixed width cubes (unit vs wide): tests interaction of very general and very specific terms.
MIXED_WIDTH = [
    (1.0, Cube([1], [])),           # unit
    (2.0, Cube([1, 2, 3, 4], [])),  # wide
]

SIMPLE_WDNFS = [WDNF(w) for w in [
    EMPTY,
    SINGLE_POS,
    SINGLE_NEG,
    OVERLAPPING,
    UNSAT_CUBE_ONLY,
    MIXED_SAT_UNSAT,
    ZERO_WEIGHT,
    NEGATIVE_WEIGHT,
    DUP_LITERALS,
    SPARSE_INDICES,
    DUPLICATE_CUBES,
    MUTUALLY_EXCLUSIVE,
    CONST_PLUS_SPECIFIC,
    ALL_NEG,
    MIXED_WIDTH,
]]

########################################################################################
# ChatGPT complex WDNFs
########################################################################################

# 1) At-least-3-of-5 (vars 1..5) with a gate: require ~x6 and ~x7 as a practical “context”
#    Exercises k-of-n combinatorics, many cubes, and negative literals as gates.
ATLEAST3_OF5_GATED = [
    (1.0, Cube(list(pos), [v for v in [1,2,3,4,5] if v not in pos] + [6,7]))
    for pos in combinations([1,2,3,4,5], 3)
]

# 2) Odd parity on vars 1..4 (XOR-like): cubes cover all assignments with an odd number of True.
#    Exercises parity, exact coverage with positives/negatives.
PARITY_ODD_1_4 = []
for bits in range(16):
    trues = [i+1 for i in range(4) if (bits >> i) & 1]
    falses = [i+1 for i in range(4) if not ((bits >> i) & 1)]
    if (len(trues) % 2) == 1:
        PARITY_ODD_1_4.append((1.0, Cube(trues, falses)))

# 3) Hierarchical / gated by control x10:
#    If x10 then need (x1 & x2 & ~x3) OR (x4 & ~x5 & x6);
#    else need (x7 & x8) OR (~x2 & x9).
GATED_BY_X10 = [
    # x10 branch
    (2.0, Cube([10, 1, 2], [3])),
    (2.0, Cube([10, 4, 6], [5])),
    # ~x10 branch
    (1.5, Cube([7, 8], [10])),
    (1.5, Cube([9], [2, 10])),
]

# 4) Ladder of mutually exclusive “wins”: x1 & ~x2, x2 & ~x3, ..., x6 & ~x7 with ascending weights.
#    Exercises exclusivity and priority-like weighting.
LADDER_EXCLUSIVE = [
    (w, Cube([i], [i+1])) for w, i in zip([1,2,3,4,5,6], [1,2,3,4,5,6])
]

# 5) Long mixed-width interactions (wide conjunctions with both signs).
#    Exercises evaluation cost and correctness under many literals in a cube.
LONG_MIXED = [
    (3.0, Cube([1,2,3,4,5,6], [7,8])),         # very specific
    (1.0, Cube([1,2], [8])),                    # subsuming/overlapping
    (2.0, Cube([3,4,9], [2,7])),                # cross-over overlaps
    (0.5, Cube([], [10]))                       # soft global penalty when x10 is False
]

# 6) Duplicate / subsuming cubes with different weights.
#    Exercises aggregation: identical conditions should sum; strict cube should only add when satisfied.
DUP_AND_SUBSUME = [
    (1.0, Cube([1,2,3], [])),
    (2.5, Cube([1,2,3], [])),     # duplicate of the first
    (1.0, Cube([1,2,3,4], [])),   # stricter, subsumed
    (1.5, Cube([1,2], [5]))       # looser but gated by ~x5
]

# 7) Group interaction: one from A={1,2,3} AND two from B={4,5,6,7}.
#    Encoded as all combinations (a in A) ∧ (b1,b2 in B). Exercises many medium cubes and combinatorics.
GROUP_A1_B2 = [
    (0.8, Cube([a] + list(b_pair), []))
    for a in [1,2,3] for b_pair in combinations([4,5,6,7], 2)
]

# 8) Sparse high-index interactions (focus on 8..10) plus small-index gates.
#    Exercises no assumptions about contiguous/low indices and mixed signs.
SPARSE_HIGH = [
    (2.0, Cube([8,10], [9])),
    (1.0, Cube([9], [8])),
    (1.5, Cube([8,9], [])),
    (0.7, Cube([], [1,2]))     # background gate on small vars
]

# 9) Nearly-tautological backbone with exceptions:
#    A constant term plus “exception” cubes that add/subtract when certain specific patterns hold.
#    Exercises constant shift + corrective patterns including negative weights.
BACKBONE_WITH_EXCEPTIONS = [
    (5.0, Cube([], [])),                      # constant baseline
    (1.0, Cube([1,2,3], [])),                 # bonus when a strong triad holds
    (-0.5, Cube([4], [5,6])),                 # penalty when x4 alone w/ ~x5 ~x6
    (0.75, Cube([7,8], [3])),                 # cross-group interaction
    (-1.25, Cube([2,9], [1,10]))              # negative correction with specific gate
]

# 10) Robustness to contradictory + overshadowing heavy cube:
#     A heavy, specific cube dominates; some unsat cubes shouldn’t contribute.
ROBUST_DOMINANCE = [
    (10.0, Cube([1,2,3,4,5], [6,7])),
    (3.0,  Cube([1], [1])),      # unsatisfiable
    (2.0,  Cube([2,3], [8])),
    (1.0,  Cube([5,9], [])),
]

COMPLEX_WDNFS = [
    WDNF(ATLEAST3_OF5_GATED),       # k-of-n with a gate (~x6, ~x7)
    WDNF(PARITY_ODD_1_4),           # odd parity across 1..4
    WDNF(GATED_BY_X10),             # control-variable gating (x10)
    WDNF(LADDER_EXCLUSIVE),         # mutually exclusive ladder patterns
    WDNF(LONG_MIXED),               # long mixed-sign interactions
    WDNF(DUP_AND_SUBSUME),          # duplicate + subsuming cubes
    WDNF(GROUP_A1_B2),              # one-from-A and two-from-B combinations
    WDNF(SPARSE_HIGH),              # focus on high indices (8..10) with small-index gates
    WDNF(BACKBONE_WITH_EXCEPTIONS), # constant backbone with positive/negative exceptions
    WDNF(ROBUST_DOMINANCE),         # heavy specific cube + contradictory noise
]

def unite_all_wdnfs(wdnfs):
    wdnf = WDNF([])
    for other_wdnf in wdnfs:
        wdnf.extend(other_wdnf)
    return wdnf

UNITED_SIMPLED_WDNFS = unite_all_wdnfs(SIMPLE_WDNFS)
UNITED_COMPLEX_WDNFS = unite_all_wdnfs(COMPLEX_WDNFS)
UNITED_WDNFS = [UNITED_SIMPLED_WDNFS, UNITED_COMPLEX_WDNFS]

ALL_WDNFs = MY_WDNFs + SIMPLE_WDNFS + COMPLEX_WDNFS + UNITED_WDNFS
