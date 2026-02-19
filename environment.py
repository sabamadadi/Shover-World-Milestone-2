import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


try:
    import gym  
    from gym import spaces  
except Exception:
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception:
        class _DummySpace:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _SpaceDiscrete(_DummySpace):
            pass

        class _SpaceMultiDiscrete(_DummySpace):
            pass

        class _SpaceBox(_DummySpace):
            pass

        class _SpaceDict(_DummySpace):
            pass

        class _SpaceTuple(_DummySpace):
            pass

        class _Spaces:
            Discrete = _SpaceDiscrete
            MultiDiscrete = _SpaceMultiDiscrete
            Box = _SpaceBox
            Dict = _SpaceDict
            Tuple = _SpaceTuple

        spaces = _Spaces()

        class _Gym:
            class Env:
                def close(self):
                    return None

        gym = _Gym()
LAVA = -100
EMPTY = 0
BARRIER = 100
BOX_MIN = 1
BOX_MAX = 10

DIRECTION_VECS = {
    1: (-1, 0),
    2: (0, 1),
    3: (1, 0),
    4: (0, -1),
}
DIR_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3}


@dataclass
class PerfectSquareInfo:
    n: int
    top: int
    left: int
    created_at: int
    age: int = 0


class ShoverWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        n_rows: int = 9,
        n_cols: int = 13,
        max_timestep: int = 400,
        number_of_boxes: int = 15,
        number_of_barriers: int = 5,
        number_of_lavas: int = 5,
        initial_stamina: float = 1000.0,
        initial_force: float = 30.0,
        unit_force: float = 10.0,
        perf_sq_initial_age: int = 20,
        map_path: Optional[str] = None,
        seed: Optional[int] = None,
        r_step: float = -1.0,
        r_invalid: float = -5.0,
        r_lava: Optional[float] = None,
        barrier_reward_coef: float = 10.0,
        edge_lava: bool = True,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_timestep = max_timestep
        self.number_of_boxes = number_of_boxes
        self.number_of_barriers = number_of_barriers
        self.number_of_lavas = number_of_lavas

        self.initial_stamina = float(initial_stamina)
        self.initial_force = float(initial_force)
        self.unit_force = float(unit_force)
        self.perf_sq_initial_age = int(perf_sq_initial_age)
        self.map_path = map_path
        self.seed_value = seed
        self.edge_lava = bool(edge_lava)

        self.r_step = float(r_step)
        self.r_invalid = float(r_invalid)
        self.r_lava = float(initial_force) if r_lava is None else float(r_lava)
        self.barrier_reward_coef = float(barrier_reward_coef)

        self._rng = np.random.default_rng(seed)

        self.grid: np.ndarray = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        self.agent_row: int = 0
        self.agent_col: int = 0
        self.agent_dir: int = 1
        self.stamina: float = self.initial_stamina
        self.timestep: int = 0
        self.total_destroyed: int = 0

        self.non_stationary_until = np.zeros(
            (4, self.n_rows, self.n_cols), dtype=np.int32
        )
        self.perfect_squares: Dict[Tuple[int, int, int], PerfectSquareInfo] = {}

        self.prev_selected_row: int = 0
        self.prev_selected_col: int = 0
        self.prev_action: int = 0
        self.last_action_valid: bool = True

        self.action_space = spaces.Tuple(
            (
                spaces.MultiDiscrete([self.n_rows, self.n_cols]),
                spaces.Discrete(6),
            )
        )
        self._build_observation_space()

        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._cell_size = 48
        self._hud_height = 60
        self._images_loaded = False

        self._last_step_walltime = None

        self.reset()

    def _build_observation_space(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=LAVA,
                    high=BARRIER,
                    shape=(self.n_rows, self.n_cols),
                    dtype=np.int32,
                ),
                "agent": spaces.Box(
                    low=np.array([0, 0, 1], dtype=np.int32),
                    high=np.array(
                        [self.n_rows - 1, self.n_cols - 1, 4], dtype=np.int32
                    ),
                    dtype=np.int32,
                ),
                "stamina": spaces.Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array([1e9], dtype=np.float32),
                    dtype=np.float32,
                ),
                "previous_selected_position": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array(
                        [self.n_rows - 1, self.n_cols - 1], dtype=np.int32
                    ),
                    dtype=np.int32,
                ),
                "previous_action": spaces.Box(
                    low=np.array([0], dtype=np.int32),
                    high=np.array([6], dtype=np.int32),
                    dtype=np.int32,
                ),
            }
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        previous_grid = getattr(self, "grid", None)

        if self.map_path is not None:
            grid, agent_pos = self._load_map_from_file(self.map_path)
            self.grid = grid.astype(np.int32)
            self.n_rows, self.n_cols = self.grid.shape
            if agent_pos is None:
                self.agent_row, self.agent_col = self._sample_empty_cell()
            else:
                self.agent_row, self.agent_col = agent_pos

            self.action_space = spaces.Tuple(
                (
                    spaces.MultiDiscrete([self.n_rows, self.n_cols]),
                    spaces.Discrete(6),
                )
            )
            self._build_observation_space()
        else:
            new_grid = self._generate_random_map(self.n_rows, self.n_cols)
            if (
                previous_grid is not None
                and previous_grid.shape == new_grid.shape
            ):
                attempts = 0
                while np.array_equal(new_grid, previous_grid) and attempts < 5:
                    new_grid = self._generate_random_map(self.n_rows, self.n_cols)
                    attempts += 1
            self.grid = new_grid
            self.agent_row, self.agent_col = self._sample_empty_cell()

        self.timestep = 0
        self.stamina = float(self.initial_stamina)
        self._last_step_time = time.perf_counter()
        self.total_destroyed = 0
        self.agent_dir = 1
        self.prev_selected_row = 0
        self.prev_selected_col = 0
        self.prev_action = 0
        self.last_action_valid = True

        self.non_stationary_until = np.zeros(
            (4, self.n_rows, self.n_cols), dtype=np.int32
        )
        self.perfect_squares = {}
        self._update_perfect_squares(initial=True)

        self._last_step_walltime = None

        return self._get_obs()

    def step(self, action):

        now = time.perf_counter()
        delay_penalty = 0
        delay_seconds = 0.0
        if getattr(self, "_last_step_walltime", None) is not None:
            delay_seconds = now - self._last_step_walltime
            if delay_seconds > 0:
                delay_penalty = int(delay_seconds / 0.2)
                if delay_penalty > 0:
                    self.stamina -= float(delay_penalty)
        self._last_step_walltime = now

        self.timestep += 1
        reward = self.r_step

        chain_length = 0
        lava_destroyed = 0
        initial_force_charged = False
        special_dissolved: List[Tuple[int, int, int]] = []

        (row, col), action_type = self._normalize_action(action)
        self.prev_selected_row = row
        self.prev_selected_col = col
        self.prev_action = action_type
        self.last_action_valid = True

        if 1 <= action_type <= 4:
            self.agent_dir = action_type
            dr, dc = DIRECTION_VECS[action_type]


            if self._in_bounds(row, col) and self._is_box_cell(row, col):
                tr = row - dr
                tc = col - dc
                if self._in_bounds(tr, tc):
                    self.agent_row = tr
                    self.agent_col = tc

            nr = self.agent_row + dr
            nc = self.agent_col + dc

            if not self._in_bounds(nr, nc) or self.grid[nr, nc] in (BARRIER, LAVA):
                self.last_action_valid = False
                reward += self.r_invalid
            elif self._is_box_cell(nr, nc):
                (
                    moved,
                    chain_length,
                    lava_destroyed,
                    push_cost,
                    initial_force_charged,
                ) = self._push_chain(dr, dc, action_type)
                if not moved:
                    self.last_action_valid = False
                    reward += self.r_invalid
                else:
                    if lava_destroyed > 0:
                        reward += self.r_lava * lava_destroyed
            else:
                self.agent_row = nr
                self.agent_col = nc

        elif action_type == 5:
            ok, stamina_gain, rew_gain, sq_key = self._apply_barrier_maker()
            if not ok:
                self.last_action_valid = False
                reward += self.r_invalid
            else:
                reward += rew_gain
                if sq_key is not None:
                    special_dissolved.append(sq_key)

        elif action_type == 6:
            ok, destroyed, sq_key = self._apply_hellify()
            if not ok:
                self.last_action_valid = False
                reward += self.r_invalid
            else:
                if sq_key is not None:
                    special_dissolved.append(sq_key)
        else:
            self.last_action_valid = False
            reward += self.r_invalid

        auto_dissolved = self._update_perfect_squares(initial=False)

        done = self._is_terminal()

        obs = self._get_obs()
        info = self._get_info(
            chain_length=chain_length,
            initial_force_charged=initial_force_charged,
            lava_destroyed=lava_destroyed,
        )
        info["dissolved_squares"] = auto_dissolved + special_dissolved
        info["delay_penalty"] = delay_penalty
        info["delay_seconds"] = delay_seconds
        return obs, float(reward), bool(done), info

    def render(self, mode: str = "human"):
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode {mode}")

        self._ensure_pygame()
        pygame = self._pygame

        width = self.n_cols * self._cell_size
        height = self.n_rows * self._cell_size + self._hud_height
        if self._screen is None:
            self._screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Shover-World (Env.render)")

        self._screen.fill((10, 10, 20))
        self._draw_grid(self._screen)

        hud_text = (
            f"t={self.timestep} stamina={self.stamina:.1f} "
            f"boxes={self._count_boxes()} "
        )
        squares = self._sorted_perfect_squares()
        if squares:
            sq_str = ", ".join([f"{n}@({r},{c})" for n, r, c in squares])
            hud_text += f"| squares: {sq_str}"
        txt_surface = self._font.render(hud_text, True, (255, 255, 255))
        self._screen.blit(txt_surface, (10, 10))

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

        if mode == "rgb_array":
            arr = pygame.surfarray.array3d(self._screen)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._images_loaded = False
        super().close()

    def _sample_empty_cell(self) -> Tuple[int, int]:
        empties = np.argwhere(self.grid == EMPTY)
        if len(empties) == 0:
            raise RuntimeError("No empty cell available for agent.")
        idx = self._rng.integers(len(empties))
        r, c = empties[idx]
        return int(r), int(c)

    def _generate_random_map(self, n_rows: int, n_cols: int) -> np.ndarray:
        grid = np.zeros((n_rows, n_cols), dtype=np.int32)

        def place(value: int, count: int, avoid_edges_for_boxes: bool = False):
            candidates = []
            for r in range(n_rows):
                for c in range(n_cols):
                    if grid[r, c] != EMPTY:
                        continue
                    if avoid_edges_for_boxes and (
                        r == 0 or c == 0 or r == n_rows - 1 or c == n_cols - 1
                    ):
                        continue
                    candidates.append((r, c))
            if not candidates:
                return
            count_ = min(count, len(candidates))
            idxs = self._rng.choice(len(candidates), size=count_, replace=False)
            for idx in idxs:
                r, c = candidates[idx]
                grid[r, c] = value

        place(BARRIER, self.number_of_barriers, avoid_edges_for_boxes=False)
        place(LAVA, self.number_of_lavas, avoid_edges_for_boxes=False)
        place(10, self.number_of_boxes, avoid_edges_for_boxes=True)
        return grid

    @staticmethod
    def _is_box_value(v: int) -> bool:
        v = int(v)
        return BOX_MIN <= v <= BOX_MAX

    @classmethod
    def parse_map_string(cls, text: str) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip() != ""]
        if not lines:
            raise ValueError("Empty map.")

        first = lines[0].lstrip()
        if any(ch in first for ch in ".B#LA"):
            return cls._parse_symbolic(lines)
        else:
            return cls._parse_integer(lines)

    @classmethod
    def _parse_symbolic(
        cls, lines: List[str]
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        rows: List[List[int]] = []
        agent_pos: Optional[Tuple[int, int]] = None
        n_cols: Optional[int] = None

        for r, line in enumerate(lines):
            row: List[int] = []
            for ch in line:
                if ch == " ":
                    continue
                if ch == ".":
                    row.append(EMPTY)
                elif ch == "B":
                    row.append(10)
                elif ch == "#":
                    row.append(BARRIER)
                elif ch == "L":
                    row.append(LAVA)
                elif ch == "A":
                    if agent_pos is not None:
                        raise ValueError("Multiple 'A' positions.")
                    agent_pos = (r, len(row))
                    row.append(EMPTY)
                else:
                    raise ValueError(f"Unknown map symbol: {ch!r}")
            if n_cols is None:
                n_cols = len(row)
            elif len(row) != n_cols:
                raise ValueError("Inconsistent row lengths in symbolic map.")
            rows.append(row)

        grid = np.array(rows, dtype=np.int32)
        if agent_pos is None:
            raise ValueError("Symbolic map must contain exactly one 'A'.")

        cls._validate_boxes_not_on_edges(grid)
        return grid, agent_pos

    @classmethod
    def _parse_integer(
        cls, lines: List[str]
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        rows: List[List[int]] = []
        n_cols: Optional[int] = None

        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            try:
                vals = [int(tok) for tok in tokens]
            except ValueError:
                raise ValueError("Non-integer token in integer-format map.")
            if n_cols is None:
                n_cols = len(vals)
            elif len(vals) != n_cols:
                raise ValueError("Inconsistent row lengths in integer-format map.")
            for v in vals:
                if v in (EMPTY, LAVA, BARRIER) or cls._is_box_value(v):
                    continue
                raise ValueError(f"Invalid cell value: {v}")
            rows.append(vals)

        if not rows:
            raise ValueError("Empty integer-format map.")

        grid = np.array(rows, dtype=np.int32)
        cls._validate_boxes_not_on_edges(grid)
        return grid, None

    @classmethod
    def _validate_boxes_not_on_edges(cls, grid: np.ndarray) -> None:

        return

    def _load_map_from_file(
        self, path: str
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.parse_map_string(text)

    def _normalize_action(self, action) -> Tuple[Tuple[int, int], int]:
        if isinstance(action, (tuple, list)):
            if len(action) == 2:
                pos, z = action
                if isinstance(pos, (tuple, list, np.ndarray)):
                    if len(pos) != 2:
                        raise ValueError("Position must have length 2.")
                    row, col = int(pos[0]), int(pos[1])
                else:
                    raise ValueError("First element of action must be (row, col).")
                z = int(z)
            elif len(action) == 3:
                row, col, z = [int(x) for x in action]
            else:
                raise ValueError("Action must be (pos, z) or (row, col, z).")
        else:
            raise ValueError("Unsupported action type.")

        if 0 <= z <= 5:
            z = z + 1
        row = int(np.clip(row, 0, self.n_rows - 1))
        col = int(np.clip(col, 0, self.n_cols - 1))
        return (row, col), z

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _is_box_cell(self, r: int, c: int) -> bool:
        return self._is_box_value(int(self.grid[r, c]))

    def _push_chain(
        self, dr: int, dc: int, action_type: int
    ) -> Tuple[bool, int, int, float, bool]:
        dir_idx = DIR_TO_INDEX[action_type]

        r = self.agent_row + dr
        c = self.agent_col + dc
        chain: List[Tuple[int, int]] = []

        while self._in_bounds(r, c) and self._is_box_cell(r, c):
            chain.append((r, c))
            r += dr
            c += dc

        if not chain:
            return False, 0, 0, 0.0, False

        beyond_r, beyond_c = r, c
        if not self._in_bounds(beyond_r, beyond_c):
            if not self.edge_lava:
                return False, 0, 0, 0.0, False
        else:
            if self.grid[beyond_r, beyond_c] == BARRIER or self._is_box_cell(
                beyond_r, beyond_c
            ):
                return False, 0, 0, 0.0, False

        k = len(chain)
        head_r, head_c = chain[0]
        stationary = self._is_stationary(head_r, head_c, dir_idx)
        initial_force_charged = False

        push_cost = self.unit_force * k
        if stationary:
            push_cost += self.initial_force
            initial_force_charged = True

        lava_destroyed = 0
        new_positions: List[Tuple[int, int]] = []

        for (r, c) in reversed(chain):
            dest_r = r + dr
            dest_c = c + dc

            if not self._in_bounds(dest_r, dest_c):
                if self.edge_lava:
                    lava_destroyed += 1
                    self.total_destroyed += 1
                self.grid[r, c] = EMPTY
                continue

            dest_val = self.grid[dest_r, dest_c]

            if dest_val == LAVA:
                lava_destroyed += 1
                self.total_destroyed += 1
                self.grid[r, c] = EMPTY
            elif dest_val == EMPTY:
                self.grid[dest_r, dest_c] = self.grid[r, c]
                new_positions.append((dest_r, dest_c))
                self.grid[r, c] = EMPTY
            else:
                return False, 0, 0, 0.0, False

        self.agent_row += dr
        self.agent_col += dc

        expiry = self.timestep + 1
        for (nr, nc) in new_positions:
            self.non_stationary_until[dir_idx, nr, nc] = expiry

        self.stamina -= push_cost
        if lava_destroyed > 0:
            self.stamina += self.initial_force * lava_destroyed

        return True, k, lava_destroyed, push_cost, initial_force_charged

    def _is_stationary(self, r: int, c: int, dir_idx: int) -> bool:
        expiry = self.non_stationary_until[dir_idx, r, c]
        non_stationary_now = expiry == self.timestep
        return not non_stationary_now

    def _detect_perfect_squares(self) -> List[Tuple[int, int, int]]:
        squares: List[Tuple[int, int, int]] = []
        max_n = min(self.n_rows, self.n_cols)
        for n in range(2, max_n + 1):
            for top in range(self.n_rows - n + 1):
                for left in range(self.n_cols - n + 1):
                    region = self.grid[top : top + n, left : left + n]
                    if not np.all(
                        [self._is_box_value(int(v)) for v in region.reshape(-1)]
                    ):
                        continue
                    if not self._perimeter_has_no_boxes(top, left, n):
                        continue
                    squares.append((n, top, left))
        return squares

    def _perimeter_has_no_boxes(self, top: int, left: int, n: int) -> bool:
        for r in range(top - 1, top + n + 1):
            for c in range(left - 1, left + n + 1):
                if not self._in_bounds(r, c):
                    continue
                if top <= r < top + n and left <= c < left + n:
                    continue
                if self._is_box_cell(r, c):
                    return False
        return True

    def _update_perfect_squares(self, initial: bool) -> List[Tuple[int, int, int]]:

        detected = self._detect_perfect_squares()
        new_dict: Dict[Tuple[int, int, int], PerfectSquareInfo] = {}

        for (n, top, left) in detected:
            key = (n, top, left)
            if key in self.perfect_squares:
                sq = self.perfect_squares[key]
                if not initial:
                    sq.age += 1
            else:
                sq = PerfectSquareInfo(n=n, top=top, left=left, created_at=self.timestep)
            new_dict[key] = sq

        self.perfect_squares = new_dict

        to_delete: List[Tuple[int, int, int]] = []
        for key, sq in list(self.perfect_squares.items()):
            if sq.age >= self.perf_sq_initial_age:
                self.grid[sq.top : sq.top + sq.n, sq.left : sq.left + sq.n] = EMPTY
                to_delete.append(key)

        for key in to_delete:
            del self.perfect_squares[key]

        return to_delete

    def _sorted_perfect_squares(self) -> List[Tuple[int, int, int]]:
        return [
            (sq.n, sq.top, sq.left)
            for sq in sorted(self.perfect_squares.values(), key=lambda s: s.created_at)
        ]

    def _get_oldest_square(self, condition) -> Optional[PerfectSquareInfo]:
        candidates = [
            sq for sq in self.perfect_squares.values() if condition(sq.n)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.created_at)

    def _apply_barrier_maker(
        self,
    ) -> Tuple[bool, float, float, Optional[Tuple[int, int, int]]]:

        sq = self._get_oldest_square(lambda n: n >= 2)
        if sq is None:
            return False, 0.0, 0.0, None

        n, top, left = sq.n, sq.top, sq.left
        self.grid[top : top + n, left : left + n] = BARRIER
        key = (n, top, left)
        if key in self.perfect_squares:
            del self.perfect_squares[key]

        stamina_gain = float(10 * n * n)
        self.stamina += stamina_gain
        reward_gain = self.barrier_reward_coef * (n * n)
        return True, stamina_gain, reward_gain, key

    def _apply_hellify(self) -> Tuple[bool, int, Optional[Tuple[int, int, int]]]:

        sq = self._get_oldest_square(lambda n: n > 2)
        if sq is None:
            return False, 0, None

        n, top, left = sq.n, sq.top, sq.left
        destroyed = 0
        for r in range(top, top + n):
            for c in range(left, left + n):
                if self._is_box_cell(r, c):
                    destroyed += 1
                if (
                    r == top
                    or r == top + n - 1
                    or c == left
                    or c == left + n - 1
                ):
                    self.grid[r, c] = EMPTY
                else:
                    self.grid[r, c] = LAVA

        self.total_destroyed += destroyed
        key = (n, top, left)
        if key in self.perfect_squares:
            del self.perfect_squares[key]
        return True, destroyed, key

    def _count_boxes(self) -> int:
        return int(
            np.logical_and(self.grid >= BOX_MIN, self.grid <= BOX_MAX).sum()
        )

    def _is_terminal(self) -> bool:
        if self._count_boxes() == 0:
            return True
        if self.stamina <= 0:
            return True
        if self.timestep >= self.max_timestep:
            return True
        return False

    def _get_obs(self) -> Dict[str, np.ndarray]:
        grid = self.grid.astype(np.int32, copy=True)
        agent = np.array(
            [self.agent_row, self.agent_col, self.agent_dir], dtype=np.int32
        )
        stamina_arr = np.array([self.stamina], dtype=np.float32)
        prev_pos = np.array(
            [self.prev_selected_row, self.prev_selected_col], dtype=np.int32
        )
        prev_action = np.array([self.prev_action], dtype=np.int32)
        return {
            "grid": grid,
            "agent": agent,
            "stamina": stamina_arr,
            "previous_selected_position": prev_pos,
            "previous_action": prev_action,
        }

    def _get_info(
        self, chain_length: int, initial_force_charged: bool, lava_destroyed: int
    ) -> Dict:
        return {
            "timestep": self.timestep,
            "stamina": self.stamina,
            "number_of_boxes": self._count_boxes(),
            "number_destroyed": self.total_destroyed,
            "last_action_valid": self.last_action_valid,
            "chain_length": chain_length,
            "initial_force_charged": initial_force_charged,
            "lava_destroyed_this_step": lava_destroyed,
            "perfect_squares_available": self._sorted_perfect_squares(),
        }

    def _ensure_pygame(self):
        if self._pygame is not None:
            return
        import pygame

        pygame.init()
        self._pygame = pygame
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("arial", 18)
        self._load_images()

    def _load_images(self):
        if self._images_loaded:
            return
        pygame = self._pygame
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")

        def load_img(name: str, color: Tuple[int, int, int]) -> "pygame.Surface":
            path = os.path.join(assets_dir, name)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(
                    img, (self._cell_size, self._cell_size)
                )
            surf = pygame.Surface((self._cell_size, self._cell_size))
            surf.fill(color)
            return surf

        self._img_space = load_img("space.png", (230, 230, 230))
        self._img_box = load_img("box.png", (181, 101, 29))
        self._img_barrier = load_img("barrier.png", (70, 70, 80))
        self._img_lava = load_img("lava.png", (255, 80, 0))
        self._img_agent = load_img("agent.png", (0, 180, 255))
        self._images_loaded = True

    def _draw_grid(self, surface):
        pygame = self._pygame
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                x = c * self._cell_size
                y = r * self._cell_size + self._hud_height
                val = self.grid[r, c]
                if val == LAVA:
                    img = self._img_lava
                elif val == BARRIER:
                    img = self._img_barrier
                elif self._is_box_cell(r, c):
                    img = self._img_box
                else:
                    img = self._img_space
                surface.blit(img, (x, y))

        for r in range(self.n_rows + 1):
            y = r * self._cell_size + self._hud_height
            pygame.draw.line(
                surface, (40, 40, 60), (0, y), (self.n_cols * self._cell_size, y)
            )
        for c in range(self.n_cols + 1):
            x = c * self._cell_size
            pygame.draw.line(
                surface,
                (40, 40, 60),
                (x, self._hud_height),
                (x, self._hud_height + self.n_rows * self._cell_size),
            )

        ax = self.agent_col * self._cell_size + self._cell_size // 2
        ay = (
            self.agent_row * self._cell_size
            + self._hud_height
            + self._cell_size // 2
        )
        pygame.draw.circle(surface, (0, 255, 255), (ax, ay), self._cell_size // 3)
