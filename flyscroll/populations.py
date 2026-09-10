"""Named cell populations that this project reads out of the connectome.

Every group below is selected by its published MaleCNS ``type`` annotation.
Nothing here prunes or rewires the graph: these are read-only labels used to
summarise activity and to define which synapses the habituation rule touches.

The biological justification for each group is written next to it. Where a
group is used as an engineering readout rather than a claim about function,
that is stated explicitly.
"""

from __future__ import annotations

# --- Sensory input -----------------------------------------------------------

# R1-R6 express Rh1 and carry the broadband achromatic (luminance) channel.
LUMINANCE_RECEPTORS = ("R1-R6",)

# R7/R8 are the chromatic receptors of each ommatidium. Pale (p) and yellow (y)
# subtypes express different rhodopsins; d is the dorsal rim, which is
# UV/polarisation sensitive. R7 subtypes are UV (Rh3/Rh4), R8p is blue (Rh5)
# and R8y is green (Rh6).
CHROMATIC_RECEPTORS = (
    "R7p", "R7y", "R7d", "R7_unclear",
    "R8p", "R8y", "R8d", "R8_unclear",
    "R7R8_unclear",
)

RECEPTORS = LUMINANCE_RECEPTORS + CHROMATIC_RECEPTORS

# Spectral channel each receptor type is driven from. Video frames are sRGB and
# carry no ultraviolet, so the UV receptors are driven from a DECLARED PROXY
# rather than a measured spectrum; see retina.py for the coefficients.
RECEPTOR_CHANNEL = {
    "R1-R6": "luminance",
    "R7p": "uv_proxy", "R7y": "uv_proxy", "R7d": "uv_proxy",
    "R7_unclear": "uv_proxy", "R7R8_unclear": "uv_proxy",
    "R8p": "blue", "R8y": "green", "R8d": "uv_proxy",
    "R8_unclear": "green",
}

# The lamina monopolar cells are graded in vivo. They receive a tonic current in
# this model so that histaminergic (inhibitory) photoreceptor input has
# something to subtract from. This is a chosen model parameter.
LAMINA = ("L1", "L2", "L3", "L5")


# --- Motion and feature detection -------------------------------------------

# T4 (ON) and T5 (OFF) are the elementary motion detectors. The four subtypes
# tile four cardinal directions of the fly's visual field: a and b are
# horizontal (progressive / regressive), c and d are vertical (upward /
# downward). A vertically scrolling feed drives c and d hardest, which is why
# they are tracked separately from a and b.
MOTION_VERTICAL = ("T4c", "T4d", "T5c", "T5d")
MOTION_HORIZONTAL = ("T4a", "T4b", "T5a", "T5b")
MOTION_UP = ("T4c", "T5c")
MOTION_DOWN = ("T4d", "T5d")
MOTION = MOTION_VERTICAL + MOTION_HORIZONTAL

# Lobula plate tangential cells integrate T4/T5 over wide fields: HS cells for
# horizontal optic flow, VS cells for vertical. A scroll gesture is a
# whole-field vertical sweep, so VS is the natural correlate of the scroll.
WIDEFIELD_HORIZONTAL = ("HSE", "HSN", "HSS")
WIDEFIELD_VERTICAL = ("VS",)

# Lobula columnar cells are feature detectors, each type tuned to a different
# visual event. These specific tunings are from the optic-glomeruli literature.
LOOMING = ("LPLC2", "LC4", "LC6")          # expanding dark objects, escape drive
SMALL_OBJECT = ("LC11", "LC18", "LC17")    # small moving objects
BAR_AND_EDGE = ("LC12", "LC15", "LC16")    # elongated / translating features
COURTSHIP_OBJECT = ("LC10a", "LC10b", "LC10c-1", "LC10c-2", "LC10d", "LC10e")

FEATURE_DETECTORS = LOOMING + SMALL_OBJECT + BAR_AND_EDGE + COURTSHIP_OBJECT


# --- Mushroom body: novelty and habituation ---------------------------------

# Kenyon cells form the sparse, high-dimensional representation of the current
# stimulus. Their combinatorial activity is what makes habituation in this model
# stimulus-specific rather than a global fatigue term.
KENYON_PREFIX = "KC"

# MBON16 is MBON-a'3ap and MBON17 is MBON-a'3m (confirmed from the MaleCNS
# instance names "MBON16(a'3ap)" and "MBON17(a'3m)"). The a'3 compartment is the
# published novelty/familiarity readout of the mushroom body: these cells respond
# strongly to a novel stimulus and their response decays over repeated or
# prolonged exposure, recovering for a different stimulus.
#   Hattori et al., "Representations of Novelty and Familiarity in a Mushroom
#   Body Compartment", Cell 169(5), 2017.
# This is the population whose decay defines "lost interest" in this project.
NOVELTY_MBON = ("MBON16", "MBON17", "MBON28")

# PPL104 is PPL1-a'3 (MaleCNS instance "PPL104(a'3)"), the dopaminergic cell
# that innervates the same compartment and gates its plasticity.
NOVELTY_DAN = ("PPL104",)

# Other well-characterised MBONs, tracked for context only. MBON11 is
# MBON-g1pedc>a/B, the broadly inhibitory output used in aversive memory.
CONTEXT_MBON = ("MBON01", "MBON05", "MBON11", "MBON20")

# PAM cluster dopaminergic cells signal reward in the horizontal MB lobes;
# PPL101 (PPL1-g1ped) is the canonical aversive-teaching cell.
REWARD_DAN = ("PAM01", "PAM04", "PAM11")
AVERSIVE_DAN = ("PPL101",)


# --- Descending output -------------------------------------------------------

# MDN is the Moonwalker Descending Neuron: its activation drives backward
# walking, i.e. the fly disengaging from and reversing away from what is in
# front of it. It is used here as the "move on to the next reel" command.
# Mapping backward walking onto a feed scroll is an ENGINEERING CHOICE.
#   Bidaye et al., "Neuronal control of Drosophila walking direction",
#   Science 344(6184), 2014.
REVERSE_DN = ("MDN",)

# DNa02 is a steering neuron whose right-minus-left activity correlates with
# turn direction; DNp09 drives forward walking and freezing; DNp20/DNpe017 are
# the visually driven descending cells DOOMFLY used as its game controller.
# They are recorded as comparison readouts and do not drive the scroll.
STEERING_DN = ("DNa02",)
FORWARD_DN = ("DNp09",)
VISUAL_DN = ("DNp20", "DNpe017")

DESCENDING = REVERSE_DN + STEERING_DN + FORWARD_DN + VISUAL_DN


# --- Grouping used for telemetry and readouts -------------------------------

# Order matters only for display. Each entry maps a readout name to the exact
# annotated types summed into it.
READOUT_GROUPS: dict[str, tuple[str, ...]] = {
    "novelty_mbon": NOVELTY_MBON,
    "novelty_dan": NOVELTY_DAN,
    "reward_dan": REWARD_DAN,
    "aversive_dan": AVERSIVE_DAN,
    "context_mbon": CONTEXT_MBON,
    "looming": LOOMING,
    "small_object": SMALL_OBJECT,
    "bar_edge": BAR_AND_EDGE,
    "courtship_object": COURTSHIP_OBJECT,
    "motion_up": MOTION_UP,
    "motion_down": MOTION_DOWN,
    "motion_horizontal": MOTION_HORIZONTAL,
    "widefield_vertical": WIDEFIELD_VERTICAL,
    "widefield_horizontal": WIDEFIELD_HORIZONTAL,
    "reverse_dn": REVERSE_DN,
    "steering_dn": STEERING_DN,
    "forward_dn": FORWARD_DN,
    "visual_dn": VISUAL_DN,
    "lamina": LAMINA,
    "luminance_receptors": LUMINANCE_RECEPTORS,
    "chromatic_receptors": CHROMATIC_RECEPTORS,
}

# Populations whose per-side split is meaningful for the telemetry display.
SIDED_READOUTS = ("reverse_dn", "steering_dn", "visual_dn", "novelty_mbon")

# Types recorded individually (small populations worth naming in the UI).
INDIVIDUAL_READOUTS = REVERSE_DN + STEERING_DN + FORWARD_DN + VISUAL_DN \
    + NOVELTY_MBON + NOVELTY_DAN + AVERSIVE_DAN + CONTEXT_MBON

# Superclasses that make up the "visual system" scale option. Used only by the
# optional reduced-graph mode, which is always labelled as a cropped subgraph.
VISUAL_SUPERCLASSES = (
    "ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
    "visual_projection_tbc",
)


def all_named_types() -> tuple[str, ...]:
    """Every explicitly named type this module reads out."""
    names: list[str] = []
    for group in READOUT_GROUPS.values():
        names.extend(group)
    names.extend(MOTION)
    return tuple(dict.fromkeys(names))
