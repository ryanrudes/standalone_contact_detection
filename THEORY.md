# Contact detection from first principles

This document builds the whole theory of contact detection from the ground up. (It is
the terse, canonical spine — the code cites its §-numbers throughout. A long-form
telling of the same construction, gentler and fully worked, lives in
[`docs/theory-long.md`](docs/theory-long.md).)
The method is deliberately Socratic: at each step we state the simplest idea that
could work, find the precise situation where it breaks, and let that failure
*derive* the next idea. Nothing is introduced because it is fashionable; every
piece earns its place by fixing a concrete hole in the piece before it.

By the end we will have arrived, by necessity, at a fairly heavy object (a
probabilistic hybrid dynamical system inferred as a Bayesian posterior over
active-constraint structures). The point of building it this way is that every
term in that mouthful will mean something obvious, because you watched it become
necessary. The final section maps the full object back down to the small, readable
program we actually implement first.

A note on what "contact detection" even is. Physics is a forward map: a world with
certain contacts produces motion. We observe only the motion (noisily), and we want
to invert the map — to recover the contacts that must have been acting. So this is
an **inverse problem**, and like all inverse problems its difficulty is set by what
is *recoverable* from the data, not by what we wish we knew. Keep that lens; it
returns at the end as the sharpest result in the whole theory.

---

## 0. The naive starting point

The simplest possible contact detector, and the one the toy script began with:

> A point is "in contact" with the floor if it is close to the floor and not moving
> much.

Measure height above the floor, measure speed, and when both are small, call it
contact. Everything below is the story of why each word in that sentence —
"the floor," "close," "not moving," "and" — is wrong or incomplete, and what each
correction teaches us.

---

## 1. First principle: contact is *relative* and *geometric*

**The fix to "the floor."** There is no privileged floor. A contact is always
between *two bodies*, and the only thing that physically matters is their
configuration *relative to each other*. The clearest way to see this: a foot
planted on a moving skateboard has an enormous velocity in the world, yet it is
unambiguously in solid contact with the deck. A detector that measures speed in the
world frame calls this "moving, therefore not in contact" and is simply wrong.

So we commit to the first principle: **every quantity is measured in the frame of
one body of the pair (the "support"), never in the world.** The static floor is just
the special case where the support happens to be a body of infinite mass that never
moves — a degenerate case, not the general one.

The geometric primitive is the **gap function** `g`: the signed distance between the
two bodies' surfaces along the contact normal, expressed in the support's frame.
`g > 0` is separation, `g = 0` is touching, `g < 0` would be interpenetration.
For a flat support this is just "height above the surface," but written generally it
is the value of a **signed-distance field (SDF)** attached to the support body, and
the contact normal is the SDF's gradient. This one generalization (plane → SDF in a
moving frame) is what lets the same machinery handle curved surfaces, tilting decks,
walls, and edges with no special cases.

This already replaces the toy script's `clearance = z - plane_height` with the
honest object: `g` = support-relative signed distance. The "height" word is fixed.

---

## 2. First principle: contact obeys a complementarity law (Signorini)

**The fix to "close."** "Close to the surface" is a fuzzy threshold; physics gives
us something exact. Two rigid bodies cannot interpenetrate, so `g ≥ 0` always. A
contact can only **push**, never pull, so the normal contact force satisfies
`λ ≥ 0`. And — this is the crucial part — you cannot simultaneously have a gap *and*
a force: if there is space between the bodies (`g > 0`) the force is zero; if there
is a force (`λ > 0`) the gap is zero. Compactly:

    g ≥ 0,   λ ≥ 0,   g · λ = 0

This is the **Signorini condition** (a *complementarity* condition: at most one of
each pair is nonzero). It is the governing law of unilateral contact, and it reframes
our whole problem. The binary "in contact?" is exactly the question *which branch of
the complementarity are we on right now* — the `g = 0, λ ≥ 0` branch (contact) or the
`g > 0, λ = 0` branch (free). Detecting contact is detecting the **active set** of a
complementarity problem.

Two immediate dividends:

- The earlier puzzle about "penetration softening" dissolves. `g < 0` is physically
  forbidden for rigid bodies, so any apparent penetration is either sensor/fit error
  (to be tolerated) or evidence of *compliance* (to be modeled — see §7). It is never
  "more contact."
- The "under-the-table" confusion was a symptom of asking one scalar (clearance) to
  enforce feasibility. The complementarity law says feasibility is a property of the
  *active set*, decided per body-pair against that pair's own surface — not a penalty
  on a single number.

---

## 3. First principle: a contact constrains *motion*, and the pattern of constraint is the contact "type"

**The fix to "not moving."** "Not moving" is only the signature of *one* kind of
contact. A box sliding across a table is in firm contact while moving fast; a
spinning top pivots in place; a wheel rolls. So "not moving" must be refined to "not
moving *in the directions the contact forbids*."

Given that a contact exists, what relative motion does it permit? The relative
velocity of the two bodies at the contact is a 6-component object — three linear,
three angular — called a **twist** (a screw). A contact removes some of these
freedoms and leaves others:

- **normal linear** velocity → making/breaking contact (must be ~0 during sustained
  contact, large during an impact);
- **tangential linear** velocity (2 components) → sliding along the surface;
- **normal angular** velocity → spinning/pivoting about the contact;
- **tangential angular** velocity (2 components) → rolling.

Each **contact mode** is then simply *which subspace of this 6D twist space the
relative motion is allowed to live in*:

| mode | allowed relative twist |
|---|---|
| static / sticking | ≈ 0 (no relative motion at all) |
| sliding | only tangential-linear |
| pivoting / twisting | only normal-angular |
| rolling | tangential-linear **coupled to** tangential-angular by `v = ω × r` |
| impact (transient) | large, decaying normal-linear |

This is where an earlier loose remark becomes precise and important: **the contact
modes are distinguished by the *correlations between channels*, not by any channel
alone.** Rolling is *defined* by tangential velocity and angular velocity being
locked together. A model that treats velocity channels as independent literally
cannot represent rolling — it can only see "some sliding and some spinning." So the
question "are these streams independent?" has a definitive answer: no, and their
dependence is not a nuisance to be calibrated away, it is the very signal that names
the contact type.

There is a beautiful duality underneath this (twist–wrench **reciprocity**): the
directions a contact lets you *move* and the directions it can *push* are orthogonal
complements — the force does no work on any allowed motion (`wᵀv = 0`). So once you
identify the motion-freedom subspace (the mode), you get the force's *direction* for
free. Hold that thought; §7 shows it also tells us the force's direction is *all* we
can get from kinematics.

The practical upshot for features: the right thing to measure is not four redundant
scalar speeds but the **relative twist** at the contact, kept as a vector so its
internal correlations survive. (And, to separate rolling from sliding rigorously, the
relevant velocity is that of the *material point currently at the contact*,
`v = v_com + ω × r` — which is why richer contact reasoning eventually needs body
pose and geometry, not just one tracked point.)

---

## 4. First principle: we never observe the truth, so we must reason probabilistically

So far the laws are exact — but we never get to evaluate them on exact data. We
observe noisy marker positions, and to get velocities we differentiate, which
amplifies noise; to get the gap we rely on an estimated surface. We cannot check
`g · λ = 0` on clean numbers. We must reason about what the *hidden* contact state
most likely was, given *noisy* evidence. That is Bayesian inference, and it forces a
**generative model**: for each hypothesis about the contact state, a probability
distribution over what we'd observe.

Concretely, for each candidate state we write down an **emission likelihood** — how
probable the observed (gap, twist) is under that state:

- **Free** (`g > 0`, no constraint): a *diffuse* distribution. The point could be at
  any height and moving any which way; nothing is pinned.
- **Contact, static**: a *sharp peak* at `g ≈ 0` and twist `≈ 0`.
- **Contact, sliding / rolling / pivoting**: a sharp peak at `g ≈ 0` with the twist
  concentrated on that mode's subspace (and diffuse off it).

The decision between states is then a **likelihood ratio**: how much better does
"contact" explain this frame than "free"? This is strictly better than the toy
script's chi-squared survival value, for a reason worth stating plainly. The
chi-squared number only ever modeled *one* hypothesis ("how surprising is this if it
were resting contact?") — a goodness-of-fit p-value. It never modeled the
alternative, so it could not actually weigh contact *against* free, and it saturated
(once residuals were tiny it pinned at ~1 regardless of how much evidence there was).
A likelihood ratio models both sides, is calibrated, and does not saturate. The
"confidence" we report becomes an honest posterior probability of contact.

Two modeling choices fall out naturally here:

- The emission for a mode is a distribution **on the twist as a whole** (with its
  correlations), not a product of independent per-channel terms — because §3 told us
  the correlations *are* the mode. Strictly, the twist lives in the Lie algebra
  `se(3)`, so the principled distribution is a concentrated Gaussian on that manifold
  (rotations don't add linearly, and comparing linear to angular units needs a chosen
  metric); the rolling mode is a *curved* constraint manifold there, not a flat
  subspace.
- The gap's emission is asymmetric and bounded: tight tolerance on the `g > 0` side
  (a real gap quickly means "free"), a little tolerance on the `g < 0` side (sensor
  and fit error), and — because §2 forbids true penetration — the contact likelihood
  must *decay* for large negative `g` rather than rewarding it. Gross penetration
  (the shoe far below the table plane) then has essentially zero contact likelihood
  and is rejected for the right reason.

---

## 5. First principle: contact persists in time — the world is a hybrid dynamical system

Deciding each frame in isolation throws away the strongest prior we have: contacts
are *temporally coherent*. A foot does not flicker on and off every millisecond.
The toy script bolted this on afterward with three hand-tuned cleanup passes
(hysteresis, gap-bridging, blip-dropping). We can do better by recognizing what
those heuristics were *approximating*.

The physical truth is that the system is a **hybrid dynamical system**: it has
continuous behavior within a mode (flight, sustained contact, sliding each follow a
smooth equation of motion) punctuated by discrete jumps between modes. Formally:

- **Flows**: smooth dynamics inside each mode.
- **Guards**: the *conditions that trigger a switch*, and they are precisely the
  zero-crossings from §2–§3. Free→contact is triggered by the gap reaching zero
  (`g → 0`). Contact→free is triggered by the *force* reaching zero (`λ → 0`) — note
  this is subtly different from the gap reopening, and under compliance it happens
  first. Stick→slip is triggered by the tangential force hitting the friction-cone
  boundary (§7).
- **Resets**: instantaneous state changes at a switch — at an impact the velocity
  jumps discontinuously (§6).

A **Hidden Markov Model** is the tractable, discretized shadow of this hybrid system:
hidden states = modes, a transition prior = the tendency to persist, emissions =
§4. Running the standard inference on it (forward–backward) yields a smoothed
posterior probability of contact at every frame — that is our calibrated confidence —
and the Viterbi algorithm yields the single most likely *contiguous* mode sequence —
that is our clean boolean segmentation. **This one mechanism replaces all three
cleanup heuristics at once**: persistence makes brief dropouts and brief blips
expensive, so they are bridged or removed automatically and for a principled reason.

Two refinements the hybrid view hands us for free:

- Because the guards are *state-dependent*, the transition prior should be too: the
  probability of free→contact should rise as the gap approaches zero, which is
  strictly more informative than a constant switch probability.
- A plain Markov prior says dwell times are memoryless, which is wrong — the chance a
  contact ends depends on how long it has lasted and how loaded it is. The honest
  version is a **semi-Markov / explicit-duration** model with a hazard rate, which is
  also the principled replacement for the toy script's hard "minimum contact
  duration."

---

## 6. First principle: the make/break instants are singular — impacts

The moments of *making* and *breaking* contact are not just ordinary frames; they
are where the most information lives, and where our smooth assumptions break.

At touchdown the relative normal velocity is arrested almost discontinuously. In the
language of §5 this is a **reset map**: the velocity jumps (`v⁺ = −e·v⁻`, where `e`
is the coefficient of restitution). The clean way to hold both sustained contact and
these jumps in one object is to treat the contact force not as a function but as a
**measure**:

    dν = λ(t) dt  +  Σ pᵢ δ(t − tᵢ)

— a smooth part (ordinary sustained force) **plus atoms** (impulses at impact
instants), with the velocity allowed to jump at those atoms. This is not decoration;
it dictates how we must process signals: the velocity is a function of bounded
variation (smooth with jumps), so a single global smoothing that forbids jumps is
*wrong at exactly the moments we most care about*.

Why impacts deserve their own first-class treatment:

- **They are the precise event timers.** The deceleration/jerk spike pins the
  touchdown *time* far more sharply than the gradual onset of the "contact" state. For
  gait, heel-strike and toe-off are impacts; this is the gold-standard event signal.
- **They momentarily reveal force and material.** The impulse equals the change in
  momentum (`∫λ dt = m·Δv`), so with mass known the impact is a force reading without
  a force plate; the velocity ratio across it estimates restitution; the sharpness
  estimates stiffness. An impact briefly *excites* the contact and exposes properties
  invisible during quiet rest.
- **They are where smoothing must be local.** We detect impacts on a lightly-filtered
  signal, ideally by fitting a parametric event template (a matched filter) rather
  than blindly differentiating noisy positions — and they are the natural place to
  *fuse* sensors, since an IMU/accelerometer measures the deceleration directly at
  high rate where differentiated mocap is hopeless.

So the mode set gains a transient "impact" state with constrained transitions
(free→impact→established, and contact→break→free), living at a finer timescale than
the sustained modes — which also lets us represent pathological **chatter** (rapid
make/break) instead of smoothing it away.

There is a fundamental latency–accuracy tradeoff hiding here, worth naming: the best
estimate of *whether* a touchdown happened uses the frames *after* it (you confirm a
landing by seeing the subsequent rest). So a real-time (causal, filtering) detector is
necessarily less certain than an offline (smoothing) one; a fixed-lag smoother buys
accuracy with a bounded delay. The same model serves all three; only the information
window changes.

---

## 7. First principle: what is even knowable? Observability, and why squishiness matters

This is the deepest result, and it decides what any detector — no matter how clever —
*can and cannot* recover from a given sensor.

From §3's reciprocity, kinematics (motion) determine the mode, and hence the
*direction* of the contact force. But the force's **magnitude** is a different kind of
quantity: it is a Lagrange multiplier, the value needed to enforce the constraint,
and it is set by the *dynamics*, not by the motion. With pure kinematics you cannot
recover it. Worse, in a **statically indeterminate** configuration — weight shared
across two feet, a table on four legs — the magnitude is not even uniquely determined
by the full rigid-body dynamics: there is an entire family of force distributions
consistent with the *identical* observed motion. The load split between your two feet
is invisible to kinematics, in principle.

State this as the theorem it is:

> Under rigid contact, force magnitude is unobservable from kinematics alone, and in
> indeterminate configurations it is unobservable even with full dynamics.

And now the punchline that retroactively justifies the entire "squishiness" thread:
**compliance is exactly the regularizer that restores observability.** If each
contact is a spring, its force is `λ = k · δ`, tied to *its own* penetration depth
`δ`. The instant the bodies are slightly compliant, the indeterminate null space
collapses — each contact's force is pinned by its own measurable deformation, so the
load split between two feet becomes individually identifiable from the two
penetrations. Material knowledge is not a luxury feature; it is the condition under
which force and loading become recoverable at all.

This reinterprets the gap channel one last time. On the `g > 0` side it is existence
evidence ("am I touching?"). On the `g < 0` side, *when compliance is known*, the
penetration depth is a **calibrated force gauge** — `λ = k·δ` (linear spring-damper)
or `λ ∝ δ^{3/2}` (Hertzian elastic contact) — turning what the toy script treated as
"error to forgive" into a loading measurement. Loaded vs. unloaded contact (a foot
bearing weight vs. grazing) then falls out for free.

Friction closes the dynamical loop. The classical law is **set-valued**: while
sticking, the tangential force can be anything inside the cone `‖λ_t‖ ≤ μ λ_n`, and
sliding begins exactly when it reaches the boundary — which is the stick→slip guard of
§5. So with the normal force from compliance, the friction cone *predicts* whether a
contact should stick or slip, and we can cross-check that prediction against the
observed kinematics (apparent sliding with no tangential force ⇒ something is wrong).
Real friction even has a hidden pre-sliding micro-displacement state (Dahl/LuGre), a
small reversible tangential give before gross slip — another latent worth modeling
when precision demands it.

A matching observability caveat governs **calibration**: a constant sensor bias and a
true constant offset are indistinguishable from a single static pose. You can only
separate them with motion that breaks the degeneracy (seeing both contact and free
phases, or varying the load to trace the penetration–force slope). This is a
**persistent-excitation** condition: a parameter is identifiable only if the
trajectory excites it. It both bounds what self-calibration can achieve and tells us
which motions to demand (or to simulate).

---

## 8. Assembling the whole object

Every principle above now composes into one coherent estimator. The hidden thing we
infer is not a bit but a **structure**:

- *which* contacts are active, over the edges of a **contact graph** whose nodes are
  bodies and whose edges are candidate body-pair contacts (person↔deck, deck↔ground,
  hand↔rail), with proximity used as a broad-phase filter so we don't test every pair;
- for each active edge, its **mode** (the twist subspace of §3);
- its **loading** (from compliance, §7) and, with multiple markers, its **contact
  region** — which is really a *pressure distribution* over a patch, whose total is
  the force and whose first moment is the center of pressure, with point/line/patch
  contact being statements about the support of that distribution.

So "richer contact information" has an exact meaning: we compute a **Bayesian
posterior over active-constraint structures (and their modes, loading, and regions),
with calibrated uncertainty.** The Signorini complementarity of §2 acts as the prior
over which structures are even legal; the hybrid dynamics of §5–§6 act as the temporal
prior and the event model; an **energy/dissipation budget** (a static contact
dissipates nothing, sliding dissipates `μλ_n‖v_slip‖`, an impact dissipates
`½m v_n²(1−e²)`) is a global consistency check linking all contacts; and the
observability/excitation limits of §7 tell us which parts of that posterior are sharp
and which are irreducibly uncertain given the sensors.

Inference is hybrid/multiple-model estimation (an HMM/particle filter over the
discrete structure, with the continuous forces and calibration parameters marginalized
where possible), and **EM** handles self-calibration — the resting bias, for instance,
is just the contact-state's mean gap, estimated by posterior-weighted responsibility,
which is principled where the toy script's quiet-frame median was circular.

That single sentence — *a probabilistic hybrid dynamical system, inferred as a
posterior over active-constraint structures on a contact graph, regularized by
complementarity, energy, and observability* — is the whole theory. Every clause was
forced on us by a concrete failure of the clause before it.

---

## 9. Generating truth: MuJoCo as the ground-truth oracle

The theory above makes claims about quantities that are, by design, hard or impossible
to observe — which makes *validation* the central practical problem. A physics
simulator solves it, because in simulation the hidden truth is not hidden from *us*,
only from the detector.

MuJoCo is especially well-suited:

- It reports the **true active set** (`mjData.contact` — which geom pairs touch), the
  **true 6D contact wrench** (`mj_contactForce`), the **true penetration**, and the
  **true relative velocities** at each contact. From these we can label every frame
  with ground-truth existence, mode (from slip/spin), loading (from normal force), and
  impact (from force atoms) — exactly the labels §3–§8 want and that real data lacks.
- Its contact model is **natively compliant** (`solref`/`solimp` set a stiffness and
  damping), so it is the natural place to *test the observability theorem of §7
  directly*: build a statically-indeterminate rig, feed the detector only noisy marker
  positions, and confirm that force/load-split is unrecoverable — then supply the
  compliance and watch it become identifiable.
- It generates the **entire edge-case taxonomy on demand**: moving-on-moving
  (person on a skateboard), rolling vs. sliding (ball vs. box), push-until-slip
  (friction-cone guard), drop tests across restitutions (impacts), grazing/unloaded
  contact, chatter, and dense multi-contact graphs.
- It supports **domain randomization** (mass, friction, stiffness, frame rate, marker
  noise and dropout), which both stresses robustness and lets us *measure* the
  excitation/identifiability conditions of §7 empirically.

The workflow is therefore: simulate → record the full physical truth → expose to the
detector only the "observable" channel (noisy marker/pose streams, as a mocap rig
would see) → score the inferred posterior against the withheld truth.

One disciplined caveat: MuJoCo's truth is truth *for MuJoCo's contact model* (soft
convex constraints, a pyramidal/elliptic friction cone — not Hertzian reality). It
validates the estimator's *logic and identifiability*, not absolute physical fidelity.
So we develop and verify against the simulator, but keep the emission and material
models **physically parameterized** (real stiffness, real friction coefficients) so
they transfer to real captures rather than overfitting the simulator's particular
numbers.

---

## 10. The pragmatic ladder — from this theory to running code

The full object of §8 is not what we build first. We build the smallest rung that is
*honest about the theory* and climb, validating each rung against MuJoCo (§9):

0. **(where the toy script is)** single point, static plane, independent channels,
   chi-squared score, morphological cleanup. Useful only as a foil.
1. **The generative HMM core.** Support-relative gap; emission likelihoods for
   `free` vs `contact` (diffuse vs. sharp peak); likelihood-ratio evidence;
   forward–backward posterior + Viterbi segmentation replacing all cleanup; EM for the
   resting bias. (§2, §4, §5.) This alone fixes the independence double-counting, the
   saturating score, the circular calibration, and the ad-hoc time heuristics.
2. **Modes.** Model the relative *twist* jointly and add static/sliding/pivoting (and
   rolling once pose+geometry are available) as twist-subspace emissions. (§3.)
3. **Impacts and events.** A transient impact mode at a finer timescale, with a
   matched-filter event detector emitting touchdown/lift-off times. (§6.)
4. **Dynamics & material.** Penetration-as-force under known compliance, loading, and
   the friction-cone stick/slip prediction. (§7.)
5. **The graph.** Body-pair contacts over a moving-frame contact graph, with the
   structure posterior and energy/balance priors. (§8.)

Each rung is independently testable, each is a faithful (if partial) shadow of the
full hybrid-system posterior, and each adds back exactly one piece the previous rung
was provably missing. That is the same discipline as this document: never add a part
until the absence of it has bitten.
