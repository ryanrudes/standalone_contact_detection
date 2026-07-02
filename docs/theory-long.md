# Contact detection from first principles  
## A rigorous but approachable construction

> The long-form telling: the same derivation as [`THEORY.md`](../THEORY.md), built slowly,
> with every step motivated and worked. **THEORY.md is canonical** — its §-numbers are what
> the code cites — and where the two disagree, THEORY.md wins. Read this version when you
> want the gentle on-ramp; read THEORY.md when you want the spine.

This document builds the theory of contact detection from the ground up.

The guiding method is simple:

1. Start with the simplest possible idea.
2. Find exactly where it fails.
3. Let that failure force the next idea.

So nothing is introduced just because it sounds advanced. Every concept has to earn its place by solving a concrete problem that the previous concept could not solve.

By the end, we will arrive at a large-sounding object:

> a probabilistic hybrid dynamical system inferred as a Bayesian posterior over active-constraint structures.

That phrase sounds like something a committee would invent to punish graduate students. But the point of building up to it slowly is that, by the time we get there, every part of it will mean something ordinary.

- **probabilistic** because measurements are noisy;
- **hybrid** because the system switches between discrete modes like free, contact, sliding, impact;
- **dynamical** because contact is not just a per-frame label, but something that evolves over time;
- **Bayesian posterior** because we infer hidden contact states from observed motion;
- **active-constraint structure** because contact means certain physical constraints are currently active.

The final section maps that full theory back down to the smaller program we would actually build first.

---

## What is contact detection?

Physics usually goes forward.

If you know the world state, the objects, their contacts, forces, masses, friction, stiffness, and so on, then physics predicts the motion.

Contact detection asks for the reverse.

We observe motion, usually noisily, and ask:

> What contacts must have been present to produce this motion?

That makes contact detection an **inverse problem**.

A forward problem looks like this:

```text
contacts + forces + geometry + dynamics → motion
```

Contact detection looks like this:

```text
observed motion → likely contacts
```

Inverse problems are harder because many hidden causes can produce similar observations. We may want to know whether a foot is bearing weight, merely grazing the ground, sliding, sticking, or impacting. But the data may only give us marker positions and velocities. So the real question is not “what do we wish we knew?” but:

> What is actually recoverable from the data?

That recoverability question returns later as the deepest part of the theory.

---

# 0. The naive starting point

The simplest possible contact detector is this:

> A point is in contact with the floor if it is close to the floor and not moving much.

For example, suppose we track a marker on a foot. We measure:

- its height above the floor;
- its speed.

Then we might write something like:

```text
if height is small and speed is small:
    contact = true
else:
    contact = false
```

This is a reasonable first attempt. It captures a familiar case: a foot planted on a static floor.

But every word in that sentence hides a problem:

> “A point is in contact with **the floor** if it is **close** to the floor and **not moving** much.”

We will fix these pieces one by one.

- “the floor” fails because contact is not always with a fixed floor;
- “close” fails because physical contact is not just proximity;
- “not moving” fails because many contacts involve motion;
- the simple “and” fails because the evidence is noisy and time-dependent.

The rest of the theory is the systematic repair of that naive detector.

---

# 1. Contact is relative and geometric

## 1.1 The problem with “the floor”

The naive detector assumes there is a special object called “the floor.” But physics does not care about floors as a privileged category.

A contact is always between **two bodies**.

Examples:

- foot on floor;
- hand on wall;
- wheel on road;
- box on conveyor belt;
- person standing on a skateboard;
- shoe brushing against a stair edge;
- object resting inside a moving container.

In all of these cases, contact is a relation between two bodies.

The important thing is not absolute motion in the world. The important thing is motion **relative to the other body**.

Consider a foot planted on a moving skateboard.

In the world frame, the foot may be moving quickly. If the skateboard rolls forward at 3 m/s, the foot also moves at 3 m/s. A detector that says “large world velocity means no contact” would say the foot is not in contact.

But that is wrong.

Relative to the skateboard deck, the foot is not moving. It is planted.

So the first correction is:

> Contact quantities should be measured in the frame of the support body, not in the world frame.

The floor is just a special case where the support body happens not to move. It is not the general case.

---

## 1.2 The gap function

Once we stop privileging the floor, we need a general way to ask:

> How far is body A from body B?

For a flat floor, this was just “height above the floor.”

But for a curved surface, tilted board, wall, edge, or moving object, “height” is no longer enough. We need a more general geometric quantity.

That quantity is the **gap function**, usually written:

```text
g
```

The gap function is the signed distance between the surfaces of the two bodies along the contact normal.

Its sign tells us the contact geometry:

```text
g > 0    separated
g = 0    touching
g < 0    interpenetrating
```

For a point above a flat floor, `g` really is just height above the floor.

But in general, `g` is not “height.” It is the value of a **signed-distance field**, or SDF, attached to the support body.

An SDF is a function that tells us the signed distance to a surface. Near the surface, its gradient gives the contact normal.

So instead of saying:

```text
clearance = z - floor_height
```

we say:

```text
g = signed distance to the support surface,
    measured in the support body's frame
```

That one change does a lot of work.

It lets the same detector handle:

- flat floors;
- ramps;
- curved supports;
- tilted boards;
- moving platforms;
- walls;
- rails;
- edges;
- body-to-body contacts.

The naive detector was secretly a detector for “point near a static plane.” The honest detector begins with “body-pair gap in a moving support frame.”

That is the first major upgrade.

---

# 2. Contact obeys a complementarity law

## 2.1 The problem with “close”

The naive detector says contact happens when a point is “close” to the surface.

But “close” is not a law of physics. It is a threshold.

Thresholds are practical, but they are not the concept. We want the physical rule that the threshold is approximating.

For rigid unilateral contact, the rule is:

1. Bodies cannot interpenetrate.
2. Contact forces can push, but not pull.
3. If the bodies are separated, there is no contact force.

Let us write these carefully.

The gap is `g`.

The normal contact force is `λ`.

Then rigid contact obeys:

```text
g ≥ 0
λ ≥ 0
g · λ = 0
```

This is called the **Signorini condition**.

It is a **complementarity condition**.

The word “complementarity” means that two nonnegative quantities cannot both be positive at the same time. At least one must be zero.

Here:

- if `g > 0`, there is a gap, so `λ = 0`;
- if `λ > 0`, there is a contact force, so `g = 0`.

In words:

> You can have separation without force, or force without separation, but not both.

This is the central law of unilateral rigid contact.

---

## 2.2 Contact detection as active-set detection

The Signorini condition tells us that contact detection is not really about asking whether a height is below a threshold.

It is about asking:

> Which branch of the complementarity law are we on?

There are two main branches:

### Free branch

```text
g > 0
λ = 0
```

The bodies are separated. There is no normal force.

### Contact branch

```text
g = 0
λ ≥ 0
```

The bodies are touching. A normal force may be present.

So the binary question “in contact or not?” becomes:

> Is the contact constraint active?

That is why contact detection is often called **active-set detection**.

A contact constraint is “active” when it is currently enforcing the no-penetration condition.

This is already more precise than the naive rule. “Close to the floor” becomes “near the active branch of a complementarity law.”

Yes, that phrase is less cute. Physics rarely wins poetry contests.

---

## 2.3 What about penetration?

The Signorini condition also clarifies a common confusion.

For rigid bodies:

```text
g < 0
```

is physically forbidden.

If we estimate `g < 0` from data, one of two things is happening:

1. **Measurement or fitting error.**  
   The bodies are rigid, but our estimated geometry, pose, or marker position is noisy.

2. **Compliance.**  
   The bodies are not perfectly rigid. They deform slightly, so an apparent penetration corresponds to compression.

This distinction matters.

In the rigid model, penetration is not “extra contact.” It is error.

In the compliant model, penetration can become meaningful because deformation can indicate force.

We return to this in §7.

For now, the key idea is:

> Contact is not “small height.” Contact is the active branch of a geometric complementarity law.

---

# 3. Contact constrains motion, and different constraints define different contact modes

## 3.1 The problem with “not moving”

The naive detector says a point is in contact when it is close to the surface and not moving.

But many real contacts involve motion.

Examples:

- a box sliding across a table;
- a wheel rolling along the ground;
- a ball spinning against a surface;
- a hand sliding along a rail;
- a foot pivoting during a turn.

All of these are contacts. Some involve large motion.

So “not moving” is too crude.

The correct statement is:

> A contact forbids motion in some directions and allows motion in others.

During sustained contact, the bodies should not move into or away from each other along the contact normal. But they may still move tangentially, rotate, roll, or spin.

So we should not ask whether the relative motion is zero.

We should ask:

> Is the relative motion consistent with the contact mode?

---

## 3.2 Relative velocity at contact

To describe motion between two rigid bodies, we need both translation and rotation.

The combined object is called a **twist**.

A twist has six components:

```text
3 linear velocity components
3 angular velocity components
```

At a contact, we care about the **relative twist** of one body with respect to the other, measured at the contact point.

Roughly:

```text
relative twist = relative linear motion + relative angular motion
```

This twist can be decomposed into physically meaningful channels:

1. **Normal linear velocity**  
   Motion into or away from the surface.

2. **Tangential linear velocity**  
   Sliding motion along the surface. There are two tangential directions.

3. **Normal angular velocity**  
   Spinning or twisting about the contact normal.

4. **Tangential angular velocity**  
   Rolling-type rotation.

A contact mode is a pattern of which components are allowed.

---

## 3.3 Contact modes as allowed motion subspaces

Different contact types correspond to different allowed relative motions.

### Static or sticking contact

No relative motion at the contact:

```text
relative twist ≈ 0
```

Example: a foot planted on the ground.

### Sliding contact

The bodies remain in normal contact, but move tangentially:

```text
normal velocity ≈ 0
tangential velocity allowed
```

Example: a box sliding across a table.

### Pivoting or spinning contact

The bodies remain in contact, but one rotates about the contact normal:

```text
normal angular velocity allowed
other relative motion constrained
```

Example: a spinning top, or a foot pivoting in place.

### Rolling contact

Rolling is more subtle.

A rolling wheel or ball has tangential motion and angular motion, but they are not independent. They are coupled by geometry.

For a rolling object:

```text
v = ω × r
```

where:

- `v` is tangential velocity;
- `ω` is angular velocity;
- `r` is the vector from the center of mass to the contact point.

Rolling is not “some translation plus some rotation.”

Rolling is a specific relationship between translation and rotation.

That matters because a detector that treats every velocity channel independently cannot truly recognize rolling. It can only say “there is some tangential motion and some rotation.”

But rolling is defined by their correlation.

So the feature should not be a bag of independent scalar speeds. The feature should be the full relative twist, kept as a vector.

---

## 3.4 Modes are patterns, not thresholds

This is a major conceptual shift.

The naive detector asks:

```text
is speed small?
```

The better detector asks:

```text
does the relative twist lie near the allowed motion set for this contact mode?
```

For static contact, the allowed set is near zero.

For sliding, the allowed set includes tangential velocity.

For rolling, the allowed set is curved because velocity and angular velocity must satisfy a geometric relationship.

This means contact modes are distinguished by **relationships among channels**, not by individual channels alone.

That is a crucial point.

If we destroy the correlations by treating all features independently, we destroy the evidence that identifies the mode.

A detector that factorizes everything into independent scalar terms is not merely approximate. It is blind to the thing it is trying to name.

Humanity survives another data modeling mistake, barely.

---

## 3.5 Twist-wrench reciprocity

There is a beautiful duality underneath contact mechanics.

A contact limits motion in some directions. It can exert forces in the complementary directions.

The motion object is a **twist**.

The force/torque object is a **wrench**.

For an ideal constraint, allowed motion does no work against the constraint force:

```text
wᵀ v = 0
```

where:

- `w` is the wrench;
- `v` is the twist.

This is called **twist-wrench reciprocity**.

It means:

> If you know what motions a contact allows, you also know the directions in which it can apply force.

But notice the word “directions.”

Kinematics can reveal the direction of the constraint force, but not necessarily its magnitude.

That limitation becomes central in §7.

For now, the practical takeaway is:

> To classify contact type, measure the full support-relative twist at the contact, not just world speed.

---

# 4. Measurements are noisy, so contact detection must be probabilistic

## 4.1 Exact laws meet noisy data

So far, the laws have been exact:

```text
g ≥ 0
λ ≥ 0
g · λ = 0
```

and contact modes define exact allowed motion patterns.

But real data is not exact.

We observe:

- noisy marker positions;
- noisy body poses;
- estimated surfaces;
- velocities obtained by differentiating positions;
- angular velocities inferred from pose changes;
- marker dropout;
- calibration bias;
- model mismatch.

Differentiation makes noise worse. Surface estimates are imperfect. Marker positions are not the true contact point.

So we cannot simply check whether:

```text
g = 0
twist = 0
```

because these quantities are never exactly true in measured data.

Instead, we need to ask:

> Given the noisy observations, which hidden contact state most likely produced them?

That is a probabilistic inference problem.

---

## 4.2 Hidden states and observations

The true contact state is hidden.

We observe something like:

```text
observed gap
observed relative twist
```

But the underlying state might be:

```text
free
static contact
sliding contact
rolling contact
pivoting contact
impact
```

For each possible hidden state, we need a model of what observations it would likely produce.

This is called an **emission likelihood**.

It answers:

> If the hidden state were this mode, how probable would the observed gap and twist be?

---

## 4.3 Emission likelihoods

Here is the basic structure.

### Free state

If the bodies are free, the gap should usually be positive, and the twist is unconstrained.

So the distribution is broad:

```text
g can vary widely
twist can vary widely
```

The free state has a diffuse likelihood.

### Static contact

If the bodies are in static contact:

```text
g ≈ 0
relative twist ≈ 0
```

So the likelihood is sharply concentrated near zero gap and zero twist.

### Sliding contact

If the bodies are sliding:

```text
g ≈ 0
normal velocity ≈ 0
tangential velocity allowed
```

The likelihood is sharp in the constrained directions and broad in the allowed tangential direction.

### Rolling contact

If the bodies are rolling:

```text
g ≈ 0
v ≈ ω × r
```

The likelihood is concentrated near the rolling constraint manifold.

### Pivoting contact

If the bodies are pivoting:

```text
g ≈ 0
normal angular motion allowed
other relative motion constrained
```

Again, the likelihood is sharp in forbidden directions and broad in allowed directions.

---

## 4.4 Contact evidence should be a likelihood ratio

The naive script used something like a goodness-of-fit score:

> How well does this frame fit resting contact?

That is useful, but incomplete.

Why?

Because it only models one hypothesis.

A frame could fit contact reasonably well, but also fit free motion reasonably well. To decide, we need to compare hypotheses.

The right object is a **likelihood ratio**:

```text
evidence for contact =
    probability of observation under contact
    divided by
    probability of observation under free
```

More generally:

```text
P(observation | contact mode)
--------------------------------
P(observation | free)
```

This asks the real question:

> Does contact explain the data better than non-contact?

That is much better than asking:

> Is this data surprising under one arbitrarily chosen contact model?

A goodness-of-fit p-value can say “this looks compatible with static contact,” but it does not say whether free motion is an even better explanation.

A likelihood ratio compares both sides.

---

## 4.5 From likelihood to posterior probability

A likelihood ratio gives evidence from the current observation.

To turn that into a probability of contact, we combine it with a prior:

```text
posterior ∝ likelihood × prior
```

In words:

> posterior belief = new evidence × previous expectation

If contacts are rare, we need stronger evidence to infer contact. If we already believed a contact was active in the previous frame, we need less evidence to maintain it.

This is where probability becomes not just convenient but necessary.

The detector should output something like:

```text
P(contact at time t | all observations)
```

not just:

```text
contact = true/false
```

A binary label can be derived later. But the uncertainty itself is valuable.

---

## 4.6 The gap likelihood must be asymmetric

The gap channel deserves special care.

For rigid contact, true penetration is impossible. But measured penetration may occur due to noise.

So a contact likelihood should behave like this:

- sharply peaked near `g = 0`;
- tolerant of small negative `g` due to noise or fitting error;
- rapidly decreasing for large positive `g`;
- also decreasing for large negative `g`, because deep penetration is not “strong contact” in a rigid model.

This last point is important.

A naive detector might treat negative gap as extra evidence of contact. But if a marker is far below the table surface, that is not strong evidence of contact. It is evidence that something is wrong: bad geometry, wrong body pair, sensor error, or a compliant model being used without modeling compliance.

So the contact likelihood should not reward arbitrary penetration.

On the positive side, a real gap quickly means free.

On the negative side, a small violation may be tolerated, but a large violation should be rejected.

---

## 4.7 The twist likelihood must keep correlations

The twist likelihood should be a distribution over the whole twist vector, not independent scalar distributions multiplied together.

Why?

Because contact modes are defined by correlations among twist components.

Rolling is the obvious example:

```text
v = ω × r
```

The evidence for rolling is not that `v` is small or `ω` is small. It is that they match.

So the emission model should preserve joint structure.

Mathematically, the twist lives in the Lie algebra `se(3)`, the tangent space of rigid-body motions. For small motions, we can often approximate it with a Gaussian in a chosen metric, but the geometry matters:

- rotations do not add exactly like ordinary vectors;
- linear and angular velocities have different units;
- rolling constraints may form curved manifolds, not simple flat subspaces.

The implementable version may use approximations, but the principle is clear:

> Model the relative twist jointly, because the dependencies are the signal.

---

# 5. Contact persists over time: the system is hybrid

## 5.1 The problem with frame-by-frame decisions

Suppose we classify each video frame independently.

Then noise can cause flicker:

```text
free, free, contact, free, contact, contact, free, contact...
```

But real contact is temporally coherent.

A foot does not usually make and break contact every millisecond. A box sliding on a table does not randomly lose contact because one frame had a noisy marker.

The toy script handled this with cleanup heuristics:

- hysteresis;
- gap bridging;
- removing short blips;
- enforcing minimum duration.

These are practical tricks. But they are approximating a deeper fact:

> Physical systems have modes that persist over time and switch only under specific conditions.

That is the essence of a **hybrid dynamical system**.

---

## 5.2 What “hybrid” means

A hybrid dynamical system has both:

1. **continuous dynamics**, and  
2. **discrete mode switches**.

For contact, the discrete modes might be:

```text
free
impact
static contact
sliding
rolling
pivoting
breaking
```

Within each mode, the system evolves continuously.

But at certain events, it switches modes.

Examples:

- free → contact;
- contact → free;
- sticking → sliding;
- sliding → sticking;
- free → impact → sustained contact;
- contact → break → free.

These switches are not arbitrary. They are triggered by physical conditions.

---

## 5.3 Flows, guards, and resets

A hybrid system is often described using three ingredients:

### Flows

Flows are the continuous dynamics inside a mode.

For example:

- while free, the object follows flight dynamics;
- while sliding, it follows constrained sliding dynamics;
- while sticking, relative velocity at the contact is zero.

### Guards

Guards are conditions that trigger mode transitions.

Examples:

- free → contact happens when the gap reaches zero:

```text
g → 0
```

- contact → free happens when the normal force goes to zero:

```text
λ → 0
```

- sticking → sliding happens when the tangential force reaches the friction cone boundary:

```text
‖λ_t‖ = μ λ_n
```

These are not arbitrary transition rules. They come from contact mechanics.

### Resets

Resets are instantaneous changes in continuous state during a transition.

The most important example is impact.

At impact, velocity can jump almost discontinuously.

For a simple normal collision:

```text
v⁺ = -e v⁻
```

where:

- `v⁻` is pre-impact normal velocity;
- `v⁺` is post-impact normal velocity;
- `e` is the coefficient of restitution.

A reset map explains why impacts cannot be treated as ordinary smooth motion.

---

## 5.4 The Hidden Markov Model as a tractable shadow

The full hybrid system is rich and continuous. A practical first approximation is a **Hidden Markov Model**, or HMM.

In an HMM:

- hidden states are contact modes;
- observations are gap and twist measurements;
- emissions are the likelihoods from §4;
- transitions encode temporal persistence.

So we might have hidden states:

```text
free
static contact
sliding
impact
```

and transition probabilities like:

```text
P(contact at t | contact at t-1) is high
P(free at t | free at t-1) is high
P(contact at t | free at t-1) rises when g approaches zero
```

The HMM gives two useful outputs.

### Smoothed posterior

Using forward-backward inference, we compute:

```text
P(mode at time t | all observations)
```

This gives a calibrated probability at every frame.

### Most likely sequence

Using the Viterbi algorithm, we compute:

```text
most likely contiguous mode sequence
```

This gives a clean segmentation:

```text
free → impact → contact → sliding → free
```

The important point is that one principled mechanism replaces several ad-hoc cleanup passes.

Temporal persistence naturally discourages one-frame blips and dropouts.

A short missing contact frame inside a long contact segment is unlikely unless the evidence is overwhelming. A single noisy contact-looking frame inside a long free segment is also unlikely.

So the model bridges gaps and removes blips for a physical reason, not because we stapled a broom to the output and swept until it looked tidy.

---

## 5.5 State-dependent transitions

A basic HMM uses constant transition probabilities.

But contact transitions are not constant.

The probability of free → contact should depend on geometry.

If the gap is large, contact soon is unlikely.

If the gap is approaching zero with negative normal velocity, contact soon is likely.

So a better transition model is state-dependent:

```text
P(free → contact) increases as g → 0
P(free → impact) increases when g → 0 and normal velocity < 0
P(contact → free) increases when estimated normal load goes to zero
```

This makes the HMM closer to the true hybrid system.

---

## 5.6 Explicit duration

A plain Markov model has memoryless dwell times.

That means the probability of leaving a state does not depend on how long we have already been in it.

But real contact durations are not usually memoryless.

A planted foot has a typical duration. A sliding object tends to remain sliding until friction, geometry, or external forces change. A one-frame contact is suspicious.

The principled version is a **semi-Markov model**, also called an explicit-duration model.

Instead of saying:

```text
P(leave contact now) is constant
```

we say:

```text
P(leave contact now) depends on how long contact has lasted
```

This replaces hard-coded minimum-duration rules with a probabilistic duration model.

---

# 6. Contact creation and destruction are singular events: impacts

## 6.1 Why impacts are special

The moment when contact begins is not just another frame.

At touchdown, relative normal velocity can be arrested very quickly. That means velocity changes sharply.

A sustained contact might satisfy:

```text
normal relative velocity ≈ 0
```

But an impact involves:

```text
large incoming normal velocity
rapid deceleration
possible rebound
```

If we smooth the whole trajectory too aggressively, we can erase the very signal that tells us when contact happened.

This is a classic punishment for pretending the world is differentiable everywhere. The universe, ever petty, contains collisions.

---

## 6.2 Contact force as a measure

For sustained contact, we often describe the normal force as a function of time:

```text
λ(t)
```

But at impact, the force may act over a very short time with a very large magnitude. The meaningful quantity is the impulse:

```text
p = ∫ λ(t) dt
```

A clean mathematical way to include both sustained forces and impacts is to treat contact force as a **measure**:

```text
dν = λ(t) dt + Σ pᵢ δ(t - tᵢ)
```

This says:

- there is a smooth force part, `λ(t) dt`;
- plus impulse atoms, `pᵢ`, at impact times `tᵢ`.

That notation may look intimidating, but the idea is simple:

> Most of the time, contact force is spread over time. At impacts, some force is concentrated at instants.

This also means velocity can have jumps.

So we should not force the entire velocity signal to be globally smooth.

---

## 6.3 Impacts are useful, not just annoying

Impacts are difficult, but they reveal information.

### They give precise event timing

A foot touchdown may be easier to locate from a sharp deceleration spike than from the gradual onset of low gap and low velocity.

For gait, events like heel strike and toe-off are often the most important outputs.

### They reveal impulse

Impulse equals change in momentum:

```text
p = m Δv
```

If mass is known and velocity change is measured, impact gives a direct estimate of impulse.

This is a kind of force information that may be inaccessible during quiet contact.

### They reveal material properties

The ratio of outgoing to incoming normal velocity estimates restitution:

```text
e = -v⁺ / v⁻
```

The sharpness of deceleration gives information about stiffness and damping.

Quiet contact may hide material properties. Impact excites the system and reveals them.

---

## 6.4 Impact detection should use local methods

Because impacts are localized in time, they should be detected locally.

A good approach is not to take a noisy position signal, differentiate it twice, and then act surprised when the result looks like modern art.

Better approaches include:

- lightly filtering the signal;
- using a matched filter for impact-like events;
- fitting a local parametric event template;
- fusing high-rate accelerometer or IMU data;
- allowing different time resolution for impact states and sustained contact states.

The model should include transient states such as:

```text
free → impact → established contact
contact → break → free
```

This lets the detector represent sharp make/break events without smoothing them away.

It also lets us represent **chatter**, rapid repeated make/break contact, as a real phenomenon rather than a nuisance to erase.

---

## 6.5 Offline, real-time, and fixed-lag detection

There is a fundamental tradeoff.

To know that touchdown happened, it helps to see what happens after touchdown.

For example:

- before touchdown, the foot is approaching the ground;
- at touchdown, there is a deceleration;
- after touchdown, the foot remains constrained.

The later frames confirm the event.

So an offline detector, which sees the whole sequence, can be more accurate than a real-time detector that only sees the past.

There are three regimes:

### Filtering

Uses only past and present observations.

```text
P(state at t | observations up to t)
```

Useful for real-time systems, but less certain.

### Smoothing

Uses past, present, and future observations.

```text
P(state at t | all observations)
```

Best for offline analysis.

### Fixed-lag smoothing

Uses observations up to a short delay after `t`.

```text
P(state at t | observations up to t + L)
```

This trades latency for accuracy.

The same probabilistic model can support all three. Only the observation window changes.

---

# 7. What is actually knowable? Observability and compliance

## 7.1 Kinematics can reveal contact mode

From motion alone, we can often infer contact existence and contact mode.

If a point remains near a surface and its relative normal velocity is near zero, contact is plausible.

If the relative twist lies near a sliding subspace, sliding is plausible.

If translation and rotation satisfy a rolling relationship, rolling is plausible.

So kinematics can tell us a lot.

But there is a limit.

---

## 7.2 Kinematics usually cannot reveal force magnitude

The contact force magnitude is a **Lagrange multiplier**.

That means it is the force needed to enforce a constraint.

For example, if a foot is planted on the ground, the normal force is whatever it must be to prevent the foot from accelerating through the ground.

The contact constraint tells us:

```text
normal motion is forbidden
```

But it does not by itself tell us how large the force is.

Motion may reveal the direction of a constraint force, but not its magnitude.

This is a fundamental limitation, not an algorithmic failure.

The theorem is:

> Under rigid contact, force magnitude is not observable from kinematics alone.

In other words:

> Watching motion alone does not generally tell you how hard the contact is pushing.

---

## 7.3 Static indeterminacy makes it worse

Some systems have multiple contact forces that can balance the same motion.

Examples:

- a person standing on two feet;
- a table resting on four legs;
- a box supported by multiple contact points;
- a hand pressing against a wall while a foot pushes on the ground.

Suppose a person stands still on two feet.

The total upward normal force must balance weight:

```text
λ_left + λ_right = mg
```

But infinitely many load splits satisfy that equation:

```text
λ_left = 0.5 mg, λ_right = 0.5 mg
λ_left = 0.7 mg, λ_right = 0.3 mg
λ_left = 0.2 mg, λ_right = 0.8 mg
```

If the body is not moving, all of those load splits can produce identical observed kinematics.

So even with dynamics, the individual forces may be unidentifiable.

This is called **static indeterminacy**.

The stronger theorem is:

> In statically indeterminate rigid contact, individual contact force magnitudes are not uniquely recoverable even from full rigid-body dynamics.

That is a brutal but important result.

No clever detector can recover information that is not present in the data. The universe does not owe us observability. Rude, but consistent.

---

## 7.4 Compliance restores observability

Now the “squishiness” thread becomes central.

Rigid contact says:

```text
g = 0 during contact
```

That gives no deformation measurement.

Compliant contact says bodies deform slightly. The deformation or penetration depth can be related to force.

For a simple linear spring model:

```text
λ = k δ
```

where:

- `λ` is normal force;
- `k` is stiffness;
- `δ` is compression or penetration depth.

For a spring-damper model:

```text
λ = k δ + c δ̇
```

For Hertzian elastic contact:

```text
λ ∝ δ^(3/2)
```

Now each contact’s force is tied to its own deformation.

In the two-foot example:

```text
λ_left = k_left δ_left
λ_right = k_right δ_right
```

If we can measure or infer the two deformations and know the stiffnesses, then the load split becomes identifiable.

So compliance is not just a nuisance.

It is the regularizer that turns an unobservable rigid force distribution into an observable deformation-based force estimate.

That is the punchline:

> Squishiness makes loading visible.

---

## 7.5 The gap has two meanings

The gap channel now has different interpretations on different sides.

### Positive gap

```text
g > 0
```

This is separation evidence.

A large positive gap means no contact.

### Near-zero gap

```text
g ≈ 0
```

This is contact-existence evidence.

The bodies are touching or nearly touching.

### Negative gap under rigid modeling

```text
g < 0
```

This is usually sensor error, pose error, geometry error, or model mismatch.

### Negative gap under compliant modeling

If compliance is known, negative gap can represent deformation:

```text
δ = -g
```

Then the gap becomes a force gauge:

```text
λ = k(-g)
```

or more generally:

```text
λ = f(δ, δ̇)
```

So the same measured channel can mean different things depending on the physical model.

In a rigid model, penetration is error to tolerate carefully.

In a compliant model, penetration may be the signal that reveals loading.

---

## 7.6 Friction and stick-slip

Normal contact tells us whether bodies push into each other.

Friction governs tangential force.

The classical Coulomb friction law is set-valued.

During sticking:

```text
‖λ_t‖ ≤ μ λ_n
```

The tangential force can take whatever value is needed, up to the friction limit.

Sliding begins when the tangential force reaches the boundary:

```text
‖λ_t‖ = μ λ_n
```

Then sliding friction acts opposite slip velocity.

This gives a physical guard for stick → slip transitions:

> sticking ends when required tangential force exceeds the friction cone.

If compliance gives us `λ_n`, then the friction cone predicts whether sticking is possible. We can compare that prediction against observed motion.

For example:

- observed sliding while tangential force should be below the cone suggests wrong friction, wrong contact hypothesis, or unmodeled dynamics;
- observed sticking while predicted tangential demand exceeds the cone suggests bad force/load estimate or wrong mode.

Real friction can be even richer. Models like Dahl or LuGre include a hidden pre-sliding displacement: a small reversible tangential deformation before gross slip.

That matters when high precision is needed.

But the basic point is enough:

> Contact mode, loading, and friction are coupled. A good detector should not treat them as unrelated labels.

---

## 7.7 Calibration also has observability limits

Calibration has the same issue.

Suppose a sensor has a constant vertical bias. Suppose the true surface is also offset slightly from where the model thinks it is.

From one static pose, these are indistinguishable.

A measured gap error could be:

- marker bias;
- surface offset;
- body pose error;
- actual deformation;
- wrong geometry.

To separate these, the trajectory must contain informative variation.

For example:

- both contact and free phases;
- different loads;
- different poses;
- impacts;
- multiple support surfaces;
- known calibration motions.

This is called a **persistent-excitation** condition.

A parameter is identifiable only if the data excites the effect that parameter controls.

So self-calibration is not magic. It requires motion that breaks degeneracies.

If the data never distinguishes two explanations, no inference method can confidently choose between them.

---

# 8. The full object: a posterior over active contact structures

Now we can assemble the full theory.

The hidden thing we infer is not just a single contact bit.

It is a whole structure.

---

## 8.1 Contact graph

Imagine a graph:

- nodes are bodies;
- edges are possible contacts between body pairs.

Examples:

```text
foot ↔ ground
hand ↔ rail
box ↔ table
wheel ↔ road
person ↔ skateboard
skateboard ↔ ground
```

At any time, some edges are active and some are inactive.

The active edges form the current contact structure.

We do not test every possible pair in a large scene. We use proximity as a broad-phase filter, then evaluate candidate contacts in detail.

---

## 8.2 What lives on each edge?

For each candidate contact edge, we may infer:

### Existence

Is this contact active?

```text
free vs contact
```

### Mode

If active, what type?

```text
static
sliding
rolling
pivoting
impact
breaking
```

### Loading

How much force or pressure is present?

Rigid kinematics may not determine this. Compliance can make it observable.

### Contact region

Contact may not be a single point.

It may be:

- point contact;
- line contact;
- patch contact;
- distributed pressure over an area.

A pressure distribution has:

- total force;
- center of pressure;
- moment;
- support region.

So richer contact information means more than a binary label.

It means estimating a structured physical object.

---

## 8.3 The posterior

Because observations are noisy and some quantities are unobservable, we infer a distribution:

```text
P(contact structure, modes, loading, regions | observations)
```

This is the Bayesian posterior.

It tells us not only the most likely contact explanation, but also the uncertainty.

Some parts of the posterior may be sharp:

```text
this foot is definitely in contact
```

Some may be broad:

```text
the load split between the two feet is uncertain
```

That uncertainty is not a flaw. It is the honest answer when the sensors do not contain enough information.

---

## 8.4 Priors and constraints

The posterior is shaped by several kinds of physical structure.

### Complementarity

Signorini contact restricts which states are legal:

```text
g ≥ 0
λ ≥ 0
gλ = 0
```

### Hybrid dynamics

Modes persist and switch under guards:

```text
gap closing triggers contact
force vanishing triggers lift-off
friction cone boundary triggers slip
```

### Energy and dissipation

Contact modes have characteristic energy behavior:

- static contact dissipates no energy through relative motion;
- sliding dissipates energy roughly like:

```text
μ λ_n ‖v_slip‖
```

- impacts dissipate energy depending on restitution:

```text
1/2 m v_n² (1 - e²)
```

Energy gives a global consistency check.

If a proposed contact explanation creates energy from nowhere or dissipates absurd amounts, it should be penalized.

### Observability

The model must know when a quantity is identifiable.

It should not pretend to know force magnitudes from pure kinematics when they are structurally unobservable.

That is not humility. It is mathematical hygiene.

---

## 8.5 Inference

The full inference problem is hybrid and multiple-model.

Discrete variables:

```text
which contacts are active
which mode each contact is in
which transitions occurred
```

Continuous variables:

```text
body poses
velocities
forces
material parameters
sensor biases
calibration offsets
```

Practical inference methods may include:

- HMMs;
- switching Kalman filters;
- particle filters;
- factor graphs;
- expectation-maximization;
- variational inference;
- marginalization over continuous nuisance variables.

The toy script’s resting-bias correction can be reinterpreted here.

Instead of estimating the resting gap bias from frames we already declared to be contact, which is circular, EM estimates it using posterior responsibilities:

```text
bias estimate =
    weighted average of gap residuals,
    weighted by probability of contact
```

That is the principled version of “use likely resting frames to estimate resting offset.”

---

## 8.6 The full sentence, now decoded

We can now say the big sentence without it being empty jargon:

> Contact detection is inference in a probabilistic hybrid dynamical system, producing a Bayesian posterior over active-constraint structures on a contact graph, regularized by complementarity, energy, friction, compliance, and observability.

Translated:

> We infer which body pairs are touching, how they are touching, how those contacts evolve over time, and how certain we are, using physical laws and noisy motion data.

That is the theory.

Every part was forced by a failure of the naive detector.

---

# 9. Generating truth with simulation

## 9.1 Why validation is hard

The theory makes claims about hidden quantities:

- true contact state;
- true contact force;
- true penetration;
- true frictional state;
- impact impulses;
- load distribution;
- contact patch.

In real experiments, these are hard to observe.

Motion capture gives positions. Force plates give some forces, but only in specific setups. Pressure mats give pressure fields, but not always body-level dynamics. IMUs give acceleration but not direct contact labels.

So validating a contact detector on real data is difficult because the thing we want to compare against is often hidden.

This is where simulation is extremely useful.

---

## 9.2 MuJoCo as an oracle

In simulation, the hidden truth is available.

A simulator like MuJoCo can report:

- which geometry pairs are in contact;
- contact positions;
- contact normals;
- penetration;
- relative velocities;
- contact forces;
- friction forces;
- impulses;
- body states.

So we can run the detector as if it only had access to noisy observable channels, while secretly storing the full ground truth.

The workflow is:

```text
simulate full physical system
record hidden ground truth
hide the truth from the detector
give detector noisy marker/pose observations
infer contacts
compare inferred posterior to withheld truth
```

This is valuable because we can directly score things that real data may not label.

---

## 9.3 Testing the theory, not just the code

Simulation lets us test specific claims.

### Moving support

Person standing on a skateboard:

- world velocity is large;
- support-relative velocity is small.

A world-frame detector fails. A support-relative detector succeeds.

### Rolling vs sliding

Ball rolling versus box sliding:

- both may have tangential motion;
- rolling has the coupling `v = ω × r`;
- sliding does not.

A joint twist model should distinguish them.

### Friction cone guard

Push an object until it slips.

The transition should occur when tangential force reaches the friction limit.

### Impacts

Drop objects with different restitution and stiffness.

The detector should identify impact time, impulse, and possibly restitution-related features.

### Compliance and observability

Build a statically indeterminate setup, like a body supported by two contacts.

With rigid kinematics alone, load split should be unobservable.

With known compliance and deformation, load split should become identifiable.

That directly tests the observability theorem.

---

## 9.4 Domain randomization

Simulation can vary parameters systematically:

- mass;
- friction;
- stiffness;
- damping;
- restitution;
- geometry;
- marker noise;
- frame rate;
- camera dropout;
- pose estimation errors;
- sensor bias.

This lets us ask:

> Under what conditions does the detector remain calibrated?

It also lets us study persistent excitation.

For example:

- Which motions are needed to estimate stiffness?
- Which trajectories separate marker bias from surface offset?
- When does rolling become distinguishable from sliding?
- How much noise destroys impact timing?

Simulation gives controlled failure cases, which are often more valuable than easy successes.

---

## 9.5 A caveat about simulator truth

MuJoCo’s truth is truth for MuJoCo’s contact model.

It is not the universe handing down stone tablets.

MuJoCo uses a particular compliant contact model, friction approximation, solver, and numerical scheme. Real materials may follow Hertzian contact, plastic deformation, adhesion, anisotropic friction, or other effects.

So simulation validates:

- the estimator’s logic;
- observability claims;
- robustness;
- implementation correctness;
- sensitivity to noise and parameter variation.

It does not automatically guarantee perfect physical fidelity.

Therefore, the detector should keep material and contact models physically parameterized:

```text
stiffness
damping
friction
restitution
geometry
noise level
```

rather than overfitting arbitrary simulator-specific constants.

Simulation is the training ground, not the final court of reality. Annoying, but useful.

---

# 10. The pragmatic ladder from theory to code

The full theory is too much to build all at once.

So we build a ladder.

Each rung adds one missing piece. Each rung should be validated against simulation before moving on.

---

## Rung 0: The toy script

This is the starting point:

```text
single point
static plane
height threshold
speed threshold
independent features
chi-squared score
morphological cleanup
```

It is useful as a foil.

It shows what breaks.

But it has major problems:

- world-frame motion fails on moving supports;
- height only works for static planes;
- speed only detects static contact;
- independent channels miss rolling and other coupled modes;
- goodness-of-fit does not compare contact against free;
- cleanup heuristics are ad hoc;
- calibration is circular;
- force and loading are not addressed.

The rest of the ladder fixes these one at a time.

---

## Rung 1: Generative HMM core

Build the first honest model:

```text
support-relative gap
free/contact hidden states
emission likelihoods
likelihood ratio evidence
temporal persistence
forward-backward posterior
Viterbi segmentation
EM bias estimation
```

This already fixes many things.

### What it adds

- contact is relative to support geometry;
- observations are probabilistic;
- contact is compared against free;
- uncertainty is calibrated;
- flicker is handled by temporal inference;
- bias is estimated using posterior responsibilities.

### What it still lacks

- separate contact modes;
- twist subspace modeling;
- impacts;
- compliance;
- force/loading;
- multi-body contact graphs.

This is the smallest serious version.

---

## Rung 2: Contact modes

Add relative twist features.

Hidden states become:

```text
free
static contact
sliding
pivoting
rolling
```

Each mode has an emission likelihood over the joint twist.

The model now asks:

```text
Which motion constraint best explains the observed relative twist?
```

This fixes the “not moving” problem.

It also prevents rolling from being misread as independent sliding and spinning.

Rolling may require body pose and contact geometry, because the detector needs `r`, the vector from center of mass to contact point.

---

## Rung 3: Impacts and event timing

Add transient impact and break states:

```text
free → impact → contact
contact → break → free
```

Use local event detection:

- matched filters;
- acceleration spikes;
- velocity jumps;
- optional IMU fusion;
- finer time resolution near events.

This improves:

- touchdown timing;
- lift-off timing;
- impulse estimation;
- restitution estimation;
- chatter representation.

It also prevents smoothing from erasing the most informative part of the signal.

---

## Rung 4: Dynamics and material

Add compliance, loading, and friction.

Now the detector can estimate or reason about:

```text
penetration → force
normal load
friction cone
stick/slip guard
energy dissipation
material stiffness
damping
restitution
```

This rung addresses observability.

It distinguishes:

- contact existence;
- unloaded grazing contact;
- loaded support contact;
- rigid-model penetration error;
- compliant deformation.

This is where “squishiness” stops being an inconvenience and becomes a measurement channel.

---

## Rung 5: Contact graph and structure posterior

Finally, move from one point and one support to many bodies.

Represent possible contacts as a graph:

```text
nodes = bodies
edges = candidate contacts
```

Infer:

```text
which edges are active
which mode each active edge has
what loading each edge carries
what region or patch is involved
how the whole structure evolves
```

Add global priors:

- complementarity;
- hybrid dynamics;
- friction;
- energy consistency;
- balance;
- observability limits.

This is the full theory.

---

# 11. Summary in one line

The naive detector says:

> Contact means close to the floor and not moving.

The principled detector says:

> Contact is an inferred active constraint between body pairs, expressed through support-relative geometry and twist, evolving as a hybrid dynamical system, estimated probabilistically from noisy observations, and limited by what the sensors make observable.

That is a mouthful, but every piece is necessary.

- **support-relative geometry** fixes the floor assumption;
- **gap functions and complementarity** fix vague closeness;
- **relative twist and modes** fix the “not moving” assumption;
- **likelihood ratios and posteriors** fix noisy per-frame decisions;
- **hybrid dynamics and HMMs** fix temporal flicker;
- **impact states** fix singular make/break events;
- **observability theory** tells us what cannot be recovered;
- **compliance** makes loading recoverable;
- **contact graphs** generalize from one point to many bodies;
- **simulation** gives ground truth for validation.

The guiding discipline is:

> Never add machinery until the absence of that machinery causes a specific failure.

That is how the small threshold script grows, by necessity, into a physically grounded contact inference system.