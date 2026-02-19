from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random
import time
import math
import numpy as np

EMPTY = 0
BOX = 1
BARRIER = 2
LAVA = 3


DIRS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _is_box_value(v: int) -> bool:
   
    return 1 <= int(v) <= 10


def _cell_kind(v: int) -> int:

    v = int(v)
    if v == 0:
        return EMPTY
    if v == -100:  
        return LAVA
    if v == 100:  
        return BARRIER
    if _is_box_value(v):
        return BOX

    return BARRIER


def _compress_grid(grid: np.ndarray) -> Tuple[int, ...]:
    return tuple(_cell_kind(v) for v in grid.reshape(-1))


def _idx(r: int, c: int, n_cols: int) -> int:
    return r * n_cols + c


def _rc(i: int, n_cols: int) -> Tuple[int, int]:
    return divmod(i, n_cols)


def _in_bounds(r: int, c: int, n_rows: int, n_cols: int) -> bool:
    return 0 <= r < n_rows and 0 <= c < n_cols


def _count_boxes(state: Tuple[int, ...]) -> int:

    return int(sum(1 for x in state if x == BOX))


def _lava_positions(state: Tuple[int, ...], n_rows: int, n_cols: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i, v in enumerate(state):
        if v == LAVA:
            out.append(_rc(i, n_cols))
    return out


def _min_dist_box_to_sink(state: Tuple[int, ...], n_rows: int, n_cols: int, edge_lava: bool) -> int:

    boxes: List[Tuple[int, int]] = []
    for i, v in enumerate(state):
        if v == BOX:
            boxes.append(_rc(i, n_cols))
    if not boxes:
        return 0

    lava = _lava_positions(state, n_rows, n_cols)

    best = 10**9
    for (br, bc) in boxes:
        if lava:
            for (lr, lc) in lava:
                d = abs(br - lr) + abs(bc - lc)
                if d < best:
                    best = d

        if edge_lava:
         
            edge_dist = min(br, bc, (n_rows - 1 - br), (n_cols - 1 - bc))
            if edge_dist < best:
                best = edge_dist

    return int(0 if best == 10**9 else best)


def _heuristic(state: Tuple[int, ...], n_rows: int, n_cols: int, edge_lava: bool) -> float:

    b = _count_boxes(state)
    if b == 0:
        return 0.0
    d = _min_dist_box_to_sink(state, n_rows, n_cols, edge_lava=edge_lava)
    return float(1.0 * b + 0.25 * d)


def _build_dist_to_kill(
    state: Tuple[int, ...],
    n_rows: int,
    n_cols: int,
    edge_lava: bool,
) -> List[int]:

    n = n_rows * n_cols
    INF = 10**9
    dist = [INF] * n

    def is_wall(v: int) -> bool:
        return v in (BARRIER, LAVA)

    q: deque[int] = deque()


    for i, v in enumerate(state):
        if v != LAVA:
            continue
        lr, lc = _rc(i, n_cols)
        for dr, dc in DIRS:
            nr, nc = lr + dr, lc + dc
            if not _in_bounds(nr, nc, n_rows, n_cols):
                continue
            j = _idx(nr, nc, n_cols)
            if is_wall(state[j]):
                continue
            if dist[j] == INF:
                dist[j] = 0
                q.append(j)


    if edge_lava:
        for r in range(n_rows):
            for c in range(n_cols):
                if r not in (0, n_rows - 1) and c not in (0, n_cols - 1):
                    continue
                j = _idx(r, c, n_cols)
                if is_wall(state[j]):
                    continue
                if dist[j] == INF:
                    dist[j] = 0
                    q.append(j)


    while q:
        cur = q.popleft()
        cr, cc = _rc(cur, n_cols)
        base = dist[cur] + 1
        for dr, dc in DIRS:
            nr, nc = cr + dr, cc + dc
            if not _in_bounds(nr, nc, n_rows, n_cols):
                continue
            j = _idx(nr, nc, n_cols)
            if dist[j] != INF:
                continue
            if is_wall(state[j]):
                continue
            dist[j] = base
            q.append(j)

    return dist


def _heuristic_remove_one(
    state: Tuple[int, ...],
    dist_to_kill: Sequence[int],
) -> float:

    best = 10**9
    for i, v in enumerate(state):
        if v != BOX:
            continue
        d = int(dist_to_kill[i])
        if d < best:
            best = d

    if best >= 10**9:

        return float(_count_boxes(state))

    return float(best + 1)


def _chain_len(
    state: Tuple[int, ...],
    n_rows: int,
    n_cols: int,
    sel_r: int,
    sel_c: int,
    d: int,
) -> int:

    if not _in_bounds(sel_r, sel_c, n_rows, n_cols):
        return 0
    if state[_idx(sel_r, sel_c, n_cols)] != BOX:
        return 0
    dr, dc = DIRS[d]
    r, c = sel_r, sel_c
    cnt = 0
    while _in_bounds(r, c, n_rows, n_cols) and state[_idx(r, c, n_cols)] == BOX:
        cnt += 1
        r += dr
        c += dc
    return cnt


@dataclass(frozen=True)
class _Succ:
    action: Tuple[int, int, int] 
    state: Tuple[int, ...]
    removed: int
    h: float


def _apply_push(
    state: Tuple[int, ...],
    n_rows: int,
    n_cols: int,
    sel_r: int,
    sel_c: int,
    d: int,
    edge_lava: bool,
) -> Optional[Tuple[Tuple[int, ...], int]]:

    if not _in_bounds(sel_r, sel_c, n_rows, n_cols):
        return None
    start_i = _idx(sel_r, sel_c, n_cols)
    if state[start_i] != BOX:
        return None

    dr, dc = DIRS[d]


    if not _in_bounds(sel_r - dr, sel_c - dc, n_rows, n_cols):
        return None

    chain: List[int] = []
    r, c = sel_r, sel_c
    while _in_bounds(r, c, n_rows, n_cols) and state[_idx(r, c, n_cols)] == BOX:
        chain.append(_idx(r, c, n_cols))
        r += dr
        c += dc

    if not chain:
        return None

    beyond_r, beyond_c = r, c
    if not _in_bounds(beyond_r, beyond_c, n_rows, n_cols):
        if not edge_lava:
            return None
    else:
        beyond_val = state[_idx(beyond_r, beyond_c, n_cols)]
        if beyond_val in (BOX, BARRIER):
            return None

    new = list(state)
    removed = 0

    for i in reversed(chain):
        r0, c0 = _rc(i, n_cols)
        r1, c1 = r0 + dr, c0 + dc
        new[i] = EMPTY

        if not _in_bounds(r1, c1, n_rows, n_cols):
            removed += 1
            continue

        j = _idx(r1, c1, n_cols)
        if state[j] == LAVA:
            removed += 1
            continue

        if new[j] != EMPTY:
            return None

        new[j] = BOX

    return (tuple(new), removed)


def _generate_successors_astar(
    state: Tuple[int, ...],
    n_rows: int,
    n_cols: int,
    edge_lava: bool,
    beam_width: int,
    rng: random.Random,
    dist_to_kill: Sequence[int],
) -> List[_Succ]:

    cand: List[Tuple[Tuple[int, float, int, float], _Succ]] = []

    for i, v in enumerate(state):
        if v != BOX:
            continue
        r, c = _rc(i, n_cols)
        for d in range(4):
            out = _apply_push(state, n_rows, n_cols, r, c, d, edge_lava=edge_lava)
            if out is None:
                continue
            s2, removed = out
            h2 = _heuristic_remove_one(s2, dist_to_kill)


            chain = _chain_len(state, n_rows, n_cols, r, c, d)

            key = (
                0 if removed > 0 else 1,
                float(h2),
                -int(chain),
                float(rng.random()),
            )
            cand.append((key, _Succ(action=(r, c, d), state=s2, removed=removed, h=float(h2))))

    cand.sort(key=lambda x: x[0])
    return [s for _, s in cand[: max(1, int(beam_width))]]


def _generate_successors(
    state: Tuple[int, ...],
    n_rows: int,
    n_cols: int,
    edge_lava: bool,
    beam_width: int,
    rng: random.Random,
) -> List[_Succ]:

    cand: List[Tuple[Tuple[int, float, float], _Succ]] = []

    for i, v in enumerate(state):
        if v != BOX:
            continue
        r, c = _rc(i, n_cols)
        for d in range(4):
            out = _apply_push(state, n_rows, n_cols, r, c, d, edge_lava=edge_lava)
            if out is None:
                continue
            s2, removed = out
            h2 = _heuristic(s2, n_rows, n_cols, edge_lava=edge_lava)

            key = (0 if removed > 0 else 1, h2, rng.random())
            cand.append((key, _Succ(action=(r, c, d), state=s2, removed=removed, h=h2)))

    cand.sort(key=lambda x: x[0])
    return [s for _, s in cand[: max(1, int(beam_width))]]

@dataclass
class _SquareInfo:
    n: int
    top: int
    left: int
    created_at: int
    age: int = 0


def _perimeter_has_no_boxes(grid: List[int], n_rows: int, n_cols: int, top: int, left: int, n: int) -> bool:
    for rr in range(top - 1, top + n + 1):
        for cc in range(left - 1, left + n + 1):
            if not _in_bounds(rr, cc, n_rows, n_cols):
                continue
            if top <= rr < top + n and left <= cc < left + n:
                continue
            if grid[_idx(rr, cc, n_cols)] == BOX:
                return False
    return True


def _detect_perfect_squares(state: Tuple[int, ...], n_rows: int, n_cols: int) -> List[Tuple[int, int, int]]:
    squares: List[Tuple[int, int, int]] = []
    max_n = min(n_rows, n_cols)

    grid = list(state)

    for n in range(2, max_n + 1):
        for top in range(n_rows - n + 1):
            for left in range(n_cols - n + 1):
                ok = True
                for rr in range(top, top + n):
                    base = rr * n_cols
                    for cc in range(left, left + n):
                        if grid[base + cc] != BOX:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue

                if not _perimeter_has_no_boxes(grid, n_rows, n_cols, top, left, n):
                    continue

                squares.append((n, top, left))

    return squares


def _update_squares(
    state: Tuple[int, ...],
    squares: Dict[Tuple[int, int, int], _SquareInfo],
    timestep: int,
    perf_sq_initial_age: int,
    initial: bool,
) -> Tuple[Tuple[int, ...], Dict[Tuple[int, int, int], _SquareInfo]]:

    detected = _detect_perfect_squares(state, n_rows=_update_squares.n_rows, n_cols=_update_squares.n_cols)
    new_dict: Dict[Tuple[int, int, int], _SquareInfo] = {}


    for (n, top, left) in detected:
        key = (n, top, left)
        if key in squares:
            sq = squares[key]
            if not initial:
                sq.age += 1
        else:
            sq = _SquareInfo(n=n, top=top, left=left, created_at=timestep, age=0)
        new_dict[key] = sq

    squares = new_dict


    grid = list(state)
    to_delete: List[Tuple[int, int, int]] = []
    for key, sq in list(squares.items()):
        if sq.age >= perf_sq_initial_age:
            for rr in range(sq.top, sq.top + sq.n):
                base = rr * _update_squares.n_cols
                for cc in range(sq.left, sq.left + sq.n):
                    grid[base + cc] = EMPTY
            to_delete.append(key)

    for key in to_delete:
        if key in squares:
            del squares[key]

    return tuple(grid), squares



_update_squares.n_rows = 0  
_update_squares.n_cols = 0


def _oldest_square(squares: Dict[Tuple[int, int, int], _SquareInfo], cond) -> Optional[_SquareInfo]:
    cands = [sq for sq in squares.values() if cond(sq.n)]
    if not cands:
        return None

    return min(cands, key=lambda s: s.created_at)


def _apply_barrier_maker(
    state: Tuple[int, ...],
    squares: Dict[Tuple[int, int, int], _SquareInfo],
) -> Tuple[Tuple[int, ...], Dict[Tuple[int, int, int], _SquareInfo], bool]:
    sq = _oldest_square(squares, lambda n: n >= 2)
    if sq is None:
        return state, squares, False

    grid = list(state)
    for rr in range(sq.top, sq.top + sq.n):
        base = rr * _update_squares.n_cols
        for cc in range(sq.left, sq.left + sq.n):
            grid[base + cc] = BARRIER

    key = (sq.n, sq.top, sq.left)
    if key in squares:
        del squares[key]

    return tuple(grid), squares, True


def _apply_hellify(
    state: Tuple[int, ...],
    squares: Dict[Tuple[int, int, int], _SquareInfo],
) -> Tuple[Tuple[int, ...], Dict[Tuple[int, int, int], _SquareInfo], bool]:
    sq = _oldest_square(squares, lambda n: n > 2)
    if sq is None:
        return state, squares, False

    grid = list(state)
    top, left, n = sq.top, sq.left, sq.n

    for rr in range(top, top + n):
        base = rr * _update_squares.n_cols
        for cc in range(left, left + n):
            on_border = (rr == top) or (rr == top + n - 1) or (cc == left) or (cc == left + n - 1)
            grid[base + cc] = EMPTY if on_border else LAVA

    key = (sq.n, sq.top, sq.left)
    if key in squares:
        del squares[key]

    return tuple(grid), squares, True


class _Planner:
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        edge_lava: bool,
        beam_width: int,
        max_depth: int,
        max_expansions: int,
        time_limit_sec: float,
        rng: random.Random,
    ):
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.edge_lava = bool(edge_lava)
        self.beam_width = int(beam_width)
        self.max_depth = int(max_depth)
        self.max_expansions = int(max_expansions)
        self.time_limit_sec = float(time_limit_sec)
        self.rng = rng

    def plan_remove_one_astar(self, start: Tuple[int, ...]) -> List[Tuple[int, int, int]]:
        start_boxes = _count_boxes(start)
        if start_boxes == 0:
            return []

        deadline = time.perf_counter() + self.time_limit_sec

        w = 1.75


        dist_to_kill = _build_dist_to_kill(
            start,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            edge_lava=self.edge_lava,
        )


        base_depth = int(self.max_depth)
        base_exp = int(self.max_expansions)
        depth_caps = [base_depth, max(base_depth + 12, int(base_depth * 1.5)), max(base_depth + 24, int(base_depth * 2))]
        depth_caps = [min(90, d) for d in depth_caps]
        exp_caps = [base_exp, min(200000, base_exp * 2), min(250000, base_exp * 4)]

        def run_one(depth_limit: int, exp_limit: int) -> List[Tuple[int, int, int]]:
            open_heap: List[Tuple[float, int, Tuple[int, ...]]] = []
            g_score: Dict[Tuple[int, ...], int] = {start: 0}
            came_from: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], Tuple[int, int, int]]] = {}

            h0 = _heuristic_remove_one(start, dist_to_kill)
            heappush(open_heap, (w * h0, 0, start))

            expansions = 0
            while open_heap and expansions < exp_limit:
                if time.perf_counter() > deadline:
                    break

                _f, g, state = heappop(open_heap)

                if _count_boxes(state) < start_boxes:
                    return self._reconstruct(came_from, state)

                if g >= depth_limit:
                    continue

                expansions += 1

                succs = _generate_successors_astar(
                    state,
                    self.n_rows,
                    self.n_cols,
                    edge_lava=self.edge_lava,
                    beam_width=self.beam_width,
                    rng=self.rng,
                    dist_to_kill=dist_to_kill,
                )

                for s in succs:
                    nxt = s.state
                    ng = g + 1
                    if ng >= g_score.get(nxt, 10**9):
                        continue
                    g_score[nxt] = ng
                    came_from[nxt] = (state, s.action)
                    nf = ng + w * s.h
                    heappush(open_heap, (nf, ng, nxt))

            return []

        for depth_limit, exp_limit in zip(depth_caps, exp_caps):
            sub = run_one(depth_limit, exp_limit)
            if sub:
                return sub
            if time.perf_counter() > deadline:
                break

        return []

    def _reconstruct(
        self,
        came_from: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], Tuple[int, int, int]]],
        goal: Tuple[int, ...],
    ) -> List[Tuple[int, int, int]]:
        cur = goal
        rev: List[Tuple[int, int, int]] = []
        while cur in came_from:
            parent, act = came_from[cur]
            rev.append(act)
            cur = parent
        rev.reverse()
        return rev



    def plan_remove_one_rbfs(self, start: Tuple[int, ...]) -> List[Tuple[int, int, int]]:
        start_boxes = _count_boxes(start)
        if start_boxes == 0:
            return []

        self._rbfs_start_boxes = start_boxes
        self._rbfs_deadline = time.perf_counter() + self.time_limit_sec
        self._rbfs_expansions = 0
        self._rbfs_best_g: Dict[Tuple[int, ...], int] = {}

        h0 = _heuristic(start, self.n_rows, self.n_cols, edge_lava=self.edge_lava)
        f0 = h0

        path: set = {start}
        actions, _best_f = self._rbfs(start, g=0, f=f0, f_limit=float("inf"), path=path)
        return [] if actions is None else actions

    def _rbfs(
        self,
        state: Tuple[int, ...],
        g: int,
        f: float,
        f_limit: float,
        path: set,
    ) -> Tuple[Optional[List[Tuple[int, int, int]]], float]:
 
        if self._rbfs_expansions >= self.max_expansions:
            return None, float("inf")
        if time.perf_counter() > self._rbfs_deadline:
            return None, float("inf")
        if g >= self.max_depth:
            return None, float("inf")

        
        if _count_boxes(state) < self._rbfs_start_boxes:
            return [], f

  
        prev_best = self._rbfs_best_g.get(state)
        if prev_best is not None and g >= prev_best:
            return None, float("inf")
        self._rbfs_best_g[state] = g

        self._rbfs_expansions += 1

        succs = _generate_successors(
            state,
            self.n_rows,
            self.n_cols,
            edge_lava=self.edge_lava,
            beam_width=self.beam_width,
            rng=self.rng,
        )

        if not succs:
            return None, float("inf")

     
        local: List[Tuple[float, _Succ]] = []
        for s in succs:
            if s.state in path:
                continue
            g2 = g + 1
            f2 = max(float(g2 + s.h), float(f))
            local.append((f2, s))

        if not local:
            return None, float("inf")

        while True:
            local.sort(key=lambda x: x[0])
            best_f, best_s = local[0]
            if best_f > f_limit:
                return None, best_f

            alternative = local[1][0] if len(local) > 1 else float("inf")

            path.add(best_s.state)
            result, new_best_f = self._rbfs(
                best_s.state,
                g=g + 1,
                f=best_f,
                f_limit=min(f_limit, alternative),
                path=path,
            )
            path.remove(best_s.state)

            local[0] = (new_best_f, best_s)

            if result is not None:
                return [best_s.action] + result, new_best_f


class PlayerAI:
    def __init__(
        self,
        algorithm: str = "astar",
        seed: int = 0,

        max_subgoal_depth: int = 22,
        max_expansions_per_subgoal: int = 12000,
        time_limit_sec: float = 4.0,
   
        planning_time_limit_sec: Optional[float] = None,
        beam_width: int = 60,
        edge_lava: bool = True,
  
        perf_sq_initial_age: int = 20,
        prefer_hellify: bool = True,
   
        use_barrier_maker: Optional[bool] = None,
        barrier_if_low_stamina: float = 120.0,
        episode_action_budget: int = 350,
        episode_time_limit_sec: float = 40.0,
        rbfs_fallback_to_astar: bool = True,
    ):
        algo = str(algorithm).lower().strip()
        if algo not in ("astar", "rbfs"):
            raise ValueError("algorithm must be 'astar' or 'rbfs'")

        self.algorithm = algo
        self.edge_lava = bool(edge_lava)

        self.max_subgoal_depth = int(max_subgoal_depth)
        self.max_expansions_per_subgoal = int(max_expansions_per_subgoal)
        if planning_time_limit_sec is not None:
            time_limit_sec = float(planning_time_limit_sec)
        self.time_limit_sec = float(time_limit_sec)
        self.beam_width = int(beam_width)

        self.perf_sq_initial_age = int(perf_sq_initial_age)
        self.prefer_hellify = bool(prefer_hellify)
        if use_barrier_maker is None:
        
            self.use_barrier_maker = (self.algorithm != "astar")
        else:
            self.use_barrier_maker = bool(use_barrier_maker)
        self.barrier_if_low_stamina = float(barrier_if_low_stamina)

        self.episode_action_budget = int(episode_action_budget)
        self.episode_time_limit_sec = float(episode_time_limit_sec)
        self.rbfs_fallback_to_astar = bool(rbfs_fallback_to_astar)

        self._rng = random.Random(seed)

        self._planned_once = False
        self._plan: List[Tuple[Tuple[int, int], int]] = []

    def get_action(self, obs: Dict[str, Any], info: Optional[Dict[str, Any]] = None):

        if not self._planned_once:
            self._planned_once = True
            self._plan = self._plan_episode(obs)

        if self._plan:
            return self._plan.pop(0)


        grid = np.array(obs["grid"], dtype=np.int32)
        n_rows, n_cols = grid.shape
        state = _compress_grid(grid)

        if _count_boxes(state) == 0:
            return ((0, 0), 0)

        if self.algorithm == "astar":
            dist_to_kill_now = _build_dist_to_kill(state, n_rows, n_cols, edge_lava=self.edge_lava)
            succs = _generate_successors_astar(
                state,
                n_rows,
                n_cols,
                edge_lava=self.edge_lava,
                beam_width=max(1, min(self.beam_width, 40)),
                rng=self._rng,
                dist_to_kill=dist_to_kill_now,
            )
        else:
            succs = _generate_successors(
                state,
                n_rows,
                n_cols,
                edge_lava=self.edge_lava,
                beam_width=max(1, min(self.beam_width, 40)),
                rng=self._rng,
            )

        if succs:
            r, c, d = succs[0].action
            return ((int(r), int(c)), int(d))


        return ((0, 0), 0)



    def _plan_episode(self, obs: Dict[str, Any]) -> List[Tuple[Tuple[int, int], int]]:
        grid = np.array(obs["grid"], dtype=np.int32)
        n_rows, n_cols = grid.shape
        state = _compress_grid(grid)

    
        _update_squares.n_rows = int(n_rows) 
        _update_squares.n_cols = int(n_cols)  

  
        timestep = 0
        squares: Dict[Tuple[int, int, int], _SquareInfo] = {}
        state, squares = _update_squares(
            state,
            squares,
            timestep=timestep,
            perf_sq_initial_age=self.perf_sq_initial_age,
            initial=True,
        )

        stamina = float(np.array(obs.get("stamina", [1000.0]), dtype=np.float32).reshape(-1)[0])

        planner = _Planner(
            n_rows=n_rows,
            n_cols=n_cols,
            edge_lava=self.edge_lava,
            beam_width=self.beam_width,
            max_depth=self.max_subgoal_depth,
            max_expansions=self.max_expansions_per_subgoal,
            time_limit_sec=self.time_limit_sec,
            rng=self._rng,
        )

        t0 = time.perf_counter()
        plan: List[Tuple[Tuple[int, int], int]] = []

        def push_action_to_env(act: Tuple[int, int, int]) -> Tuple[Tuple[int, int], int]:
            r, c, d = act
            return ((int(r), int(c)), int(d))

        while _count_boxes(state) > 0 and len(plan) < self.episode_action_budget:
            if (time.perf_counter() - t0) > self.episode_time_limit_sec:
                break

           
            special = self._choose_special_action(squares, stamina)
            if special is not None:
     
                timestep += 1
                if special == 5:
                    state, squares, ok = _apply_hellify(state, squares)
                else:
                    state, squares, ok = _apply_barrier_maker(state, squares)

               
                plan.append(((0, 0), int(special)))

                state, squares = _update_squares(
                    state,
                    squares,
                    timestep=timestep,
                    perf_sq_initial_age=self.perf_sq_initial_age,
                    initial=False,
                )
                continue

          
            start_boxes = _count_boxes(state)

            if self.algorithm == "astar":
       
                dist_to_kill_now = _build_dist_to_kill(state, n_rows, n_cols, edge_lava=self.edge_lava)
                succs_now = _generate_successors_astar(
                    state,
                    n_rows,
                    n_cols,
                    edge_lava=self.edge_lava,
                    beam_width=max(1, min(self.beam_width, 80)),
                    rng=self._rng,
                    dist_to_kill=dist_to_kill_now,
                )

                if succs_now and succs_now[0].removed > 0:
                    sub = [succs_now[0].action]
                else:
                    sub = planner.plan_remove_one_astar(state)
            else:
                sub = planner.plan_remove_one_rbfs(state)
                if not sub and self.rbfs_fallback_to_astar:
                    sub = planner.plan_remove_one_astar(state)

            if not sub:
      
                if self.algorithm == "astar":
                    dist_to_kill_now = _build_dist_to_kill(state, n_rows, n_cols, edge_lava=self.edge_lava)
                    succs = _generate_successors_astar(
                        state,
                        n_rows,
                        n_cols,
                        edge_lava=self.edge_lava,
                        beam_width=max(1, min(self.beam_width, 80)),
                        rng=self._rng,
                        dist_to_kill=dist_to_kill_now,
                    )
                else:
                    succs = _generate_successors(
                        state,
                        n_rows,
                        n_cols,
                        edge_lava=self.edge_lava,
                        beam_width=max(1, min(self.beam_width, 60)),
                        rng=self._rng,
                    )
                if not succs:
                    break
                sub = [succs[0].action]

            for act in sub:
                if len(plan) >= self.episode_action_budget:
                    break

                timestep += 1
                out = _apply_push(state, n_rows, n_cols, act[0], act[1], act[2], edge_lava=self.edge_lava)
                if out is None:
               
                    return plan

                state, removed = out

       
                stamina -= 10.0  
                if removed > 0:
                    stamina += 40.0 * removed

                plan.append(push_action_to_env(act))

                state, squares = _update_squares(
                    state,
                    squares,
                    timestep=timestep,
                    perf_sq_initial_age=self.perf_sq_initial_age,
                    initial=False,
                )

    
                if _count_boxes(state) < start_boxes:
                    break

        return plan

    def _choose_special_action(
        self,
        squares: Dict[Tuple[int, int, int], _SquareInfo],
        stamina: float,
    ) -> Optional[int]:
  
        if not squares:
            return None

        has_hellify = _oldest_square(squares, lambda n: n > 2) is not None
        has_barrier = _oldest_square(squares, lambda n: n >= 2) is not None

        if self.prefer_hellify and has_hellify:
            return 5

  
        if self.use_barrier_maker and has_barrier:
            if stamina <= self.barrier_if_low_stamina:
                return 4

        return None