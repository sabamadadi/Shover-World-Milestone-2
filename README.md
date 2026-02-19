# 🧠 Shover World – Milestone 2

### Heuristic Search AI Agent (A* & RBFS)

This project implements a **Phase-2 heuristic search agent** for the ShoverWorld puzzle environment.

The agent solves grid-based box-removal maps by selecting a box and applying a push direction, while managing limited stamina and exploiting special mechanics such as **Hellify** and **Barrier Maker**.

<p align="center">
  <img src="demo.gif" alt="Application Demo" width="800">
</p>

The implementation supports:

* ⭐ A* Search (primary solver)
* 🔁 RBFS (Recursive Best-First Search)
* 🎯 Domain-specific heuristics
* 🔄 Sub-goal decomposition (remove at least one box per plan)
* 🧩 Perfect square detection & tactical completion

# 📌 Problem Overview

ShoverWorld is a grid-based puzzle where:

* `0` → Empty
* `1–10` → Boxes
* `100` → Barrier
* `-100` → Lava

The agent does **not move directly**.
Instead, each action selects a box cell `(r, c)` and applies:

```
((r, c), z)
```

Where:

* `z = 0,1,2,3` → Push (Up, Right, Down, Left)
* `z = 4` → Barrier Maker
* `z = 5` → Hellify

Goal: **Remove all boxes before stamina runs out.**

# 🗂 Project Structure

```
environment.py        # Full ShoverWorld environment
player_ai.py          # Heuristic search agent
run_player_ai.py      # Headless execution
watch_ai_steps.py     # Visualization viewer (pygame)
maps/
    map3.txt
    map4.txt
report.pdf            # Full technical report
demo.gif
```

Maps `map3` and `map4` are located inside the `maps/` folder.

Example map (map3 excerpt): 
Example map (map4 excerpt): 

# 🚀 Running the Agent

## Headless Mode

```
python run_player_ai.py --map maps/map3.txt --algo astar
```

Script reference: 

Options:

```
--algo astar | rbfs
--max_depth
--max_expansions
--time_limit
--beam
```

---

## Visualization Mode

```
python watch_ai_steps.py --map maps/map3.txt --algo astar
```

Viewer script: 

Controls:

* `SPACE` → Pause
* `N` → Single step
* `+/-` → Adjust speed
* `R` → Reset

---

# 🔎 Algorithm Design

## 🔹 A* Search (Primary Solver)

The evaluation function:

$f(s) = g(s) + w \cdot h(s)$

Where:

* `g(s)` → Estimated push cost
* `h(s)` → Domain-specific heuristic
* `w > 1` → Weighted A* for faster convergence

Key heuristic components:

* Distance-to-removal (grid BFS)
* Congestion penalty
* Square opportunity reward
* Immediate-removal preference

A* is the main solver for challenging boards.

## 🔹 RBFS (Baseline)

RBFS is included for comparison and memory efficiency.
It is bounded by depth, expansions, and time limits.

If RBFS struggles, it can optionally fall back to A*.

# 💡 Key Design Ideas

### 1️⃣ Sub-goal Decomposition

Instead of planning to clear the entire board:

> Plan until at least one box is removed → Execute → Replan.

This keeps search depth manageable.

### 2️⃣ Continue-Push Preference

If pushing the same moving box is beneficial, the agent prefers continuing the push to avoid repeated stationary costs.

### 3️⃣ Perfect Square Exploitation

The agent detects n×n perfect squares of boxes.

* For n ≥ 3 → Prefer Hellify
* For n = 2 → Conservative Barrier Maker

Square logic is detailed in the report 

### 4️⃣ Nearly-Complete Square Completion

If a square is missing exactly one box and can be completed in one push, the agent prioritizes completing it before running search.

### 5️⃣ Fail-Safe Mechanism

If search fails:

* Select any valid push
* Never loop indefinitely at corners
* Never freeze


# 📊 Evaluation

Tested on provided benchmark maps (map1–map5).
Key metrics:

* ✔ Solved?
* ⏱ Steps taken
* 🔋 Final stamina
* 🧠 Time to first removal

Details available in the full report: 

# 🛠 Technologies Used

* Python
* NumPy
* heapq (priority queues)
* deque (BFS)
* Pygame (visualization)
* Custom search implementation (no external planners)

# 🎯 Milestone 2 Goals Achieved

✔ Heuristic A* agent
✔ RBFS baseline
✔ Domain-informed heuristic
✔ Square detection & special action reasoning
✔ Runtime-safe bounded search
✔ Viewer + headless runner

# 📄 Report

For full algorithmic details, heuristic derivations, and experimental discussion:

See: `report.pdf` 
