# RELATE — Explained for Everyone

A friendly guide to what this project is, written so a curious kid (or anyone)
can follow along. The serious version with formulas lives in `README.md`.

---

## The Big Idea in One Sentence

**We're trying to grow a tiny pretend universe inside a computer, starting
from almost nothing, and watching what happens.**

That's it. That's the whole project.

---

## What Does the "Universe" Look Like?

Imagine a bunch of **dots** connected by **strings**.

```
    o─────o
    │     │
    o─────o
    │  ╲  │
    o─────o
```

- The **dots** are like tiny pieces of space. We call them *nodes*.
- The **strings** are connections between those pieces. We call them *edges*.
- Three dots all connected to each other make a **triangle**.

That's our whole universe. Just dots and strings. No stars, no planets, no
people — yet.

---

## How Does It Grow?

Every "tick" of the clock, our universe rolls a dice and does **one** of these
moves:

| Move | What it does | Picture |
|---|---|---|
| **expand** | Cuts a string in half and adds a new dot in the middle | `o──o`  →  `o─o─o` |
| **integrate** | Squishes a triangle into a single dot (a "particle") | `△` → `●` |
| **seed** | A pair of dots pops out and attaches to the main graph | _nothing_ → `o──o─▣` |
| **deflate** | Removes a lonely dot that has no friends | `o` → _gone_ |
| **connect** | Draws a string between two big lonely chunks | `▣  ▣` → `▣──▣` |
| **triangulate** | Closes an open shape into a triangle (ΔN=0, ΔT≥1) | `o─o─o` → `△` |

So the universe is constantly **growing, shrinking, and rearranging itself**.

---

## The Game It's Playing

Here's the cool part. The universe doesn't just pick moves randomly. It plays
a kind of **tug-of-war game** with itself, trying to be:

- 🟢 **Smooth** — not too bumpy or pointy
- 🟢 **Connected** — the dots like to talk to each other
- 🟢 **Rich** — interesting and complex, not boring
- 🟢 **The right size** — not too tiny, not too huge
- 🟢 **Triangle-ish** — triangles are the universe's favourite shape

It picks moves that score well on this game. We call this score the
**energy** of the universe. (Real physics works the same way!)

---

## What Crazy Things Actually Happen?

When we run the simulation, **things organise themselves** without anyone
telling them to. Here are the surprises:

### 1. Flat space appears 🌍

If you start with random dots and strings, after a while it stops looking
random. It looks like a **flat 2D sheet** — kind of like a piece of paper or
the surface of a balloon. We didn't program "make it flat." It just… does.

### 2. Particles are born ✨

When triangles get squished, a single new dot pops out. We can watch these
"particle births" happen and they look a lot like how matter might form in
the real universe.

### 3. Popular dots ("hubs") 🌟

A few special dots get **way more friends than everyone else**. One dot might
have 79 connections while everybody else only has 3. This is exactly how
**social networks** work too — there's always one super-popular person.
The universe figured this out by itself.

### 4. Time moves forward ⏳

The universe gets more complex and never goes back to being simple. That's
called the **arrow of time** — the same reason you can't un-scramble an egg.
We see it in the simulation about 76% of the time.

### 5. It has "moods" 🎭

Depending on the settings, the universe behaves differently:

- **Gas** — fluffy and disconnected, like clouds
- **Crystal** — stiff and orderly, like a snowflake
- **Complex** — interesting, alive-looking ← *the cool one*
- **Collapse** — squishes itself into a tiny tight ball

---

## Why Would Anyone Build This?

A few reasons:

1. **Maybe the real universe works this way.** Some physicists think space
   itself is made of tiny pieces (not smooth like it looks). This is a toy
   model to play with that idea.

2. **It's fun to watch things organise themselves.** You set up simple rules
   and complicated, beautiful patterns appear. Like ant colonies or snowflakes.

3. **It might teach us something new.** When the simulation surprises us —
   like the popular hubs showing up on their own — we learn something about
   how complexity in general works.

---

## Want to Try It Yourself?

You'll need a computer with **Python** installed (it's free). Then:

```bash
# 1. Get into the project folder
cd relate

# 2. Set up the toolbox
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Run the show!
python examples/quantum_breath.py
```

This will create a file called `quantum_breath.gif` — an actual little movie
of a universe being born. Open it and watch.

---

## Pictures You'll Find in This Folder

The project comes with a bunch of pictures from past experiments. Here's
what each one shows:

| Picture | What it is |
|---|---|
| `quantum_breath.gif` | A movie of a universe evolving step by step |
| `phase_diagram.png` | A map of all the different "moods" the universe can have |
| `hub_emergence.png` | The popular-dot effect appearing over time |
| `large_scale.png` | What happens with bigger universes |
| `golden_zone.png` | The "sweet spot" settings that make the nicest universes |
| `thermodynamics.png` | Proof that time goes forward in our universe |

---

## Fancy Words → Plain Words

Scientists love big words. Here's the cheat sheet:

| Fancy word | What it really means |
|---|---|
| **Graph** | Dots connected by strings |
| **Node** | A dot |
| **Edge** | A string between two dots |
| **Triangle** | Three dots all connected — the smallest "filled-in" shape |
| **Spectral dimension** | A way of asking "is this thing flat, line-like, or blob-like?" |
| **Curvature** | How bumpy something is |
| **Energy functional** | The scoring rules of the game |
| **Hub** | A super-popular dot with tons of friends |
| **Phase** | A "mood" of the universe |
| **Emergent** | Something cool that shows up by itself without being programmed |

---

## The Punchline

We made a tiny pretend universe out of dots and strings, gave it some simple
rules, and pressed play. Without anyone telling it to, it grew, made flat
space, made particles, made popular dots, and remembered which way time
should flow.

It's a little reminder that **simple rules can produce really beautiful,
surprising things** — which might just be how our own universe works too.
