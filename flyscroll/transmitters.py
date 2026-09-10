"""Coarse fast-transmission sign proxy for predicted neurotransmitters.

This is a declared modelling assumption, not receptor physiology. A cell's
predicted transmitter is collapsed to a single excitatory or inhibitory sign
that applies to every one of its outgoing edges. Real synapses depend on the
receptor expressed by the postsynaptic cell, which the connectome does not
report, and several of these cells co-release modulators.
"""

from __future__ import annotations

import numpy as np

EXCITATORY = frozenset({"acetylcholine"})
INHIBITORY = frozenset({"gaba", "glutamate", "histamine"})


def transmitter_signs(transmitters, ambiguous_sign: int = 1):
    """Return (signs, uncertain) arrays; never drops an edge.

    Acetylcholine is treated as +1. GABA, glutamate and histamine are treated
    as -1. Glutamate is inhibitory at the fly's GluCl channels in the circuits
    modelled here, which is why it appears in the inhibitory set. A cell whose
    prediction is missing, modulator-only, or a conflicting co-transmitter
    combination keeps ``ambiguous_sign`` and is reported as uncertain.
    """
    if ambiguous_sign not in (-1, 1):
        raise ValueError("Ambiguous cells must keep an explicit sign of +1 or -1.")
    signs = np.empty(len(transmitters), dtype=np.int8)
    uncertain = np.zeros(len(transmitters), dtype=bool)
    for position, value in enumerate(transmitters):
        tokens = {token.strip() for token in str(value).lower().split(",")}
        fast = set()
        if tokens & EXCITATORY:
            fast.add(1)
        if tokens & INHIBITORY:
            fast.add(-1)
        if len(fast) == 1:
            signs[position] = next(iter(fast))
        else:
            signs[position] = ambiguous_sign
            uncertain[position] = True
    return signs, uncertain
