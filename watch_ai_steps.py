from __future__ import annotations
import argparse
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
from environment import ShoverWorldEnv, LAVA, BARRIER, BOX_MIN, BOX_MAX
from player_ai import PlayerAI

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=str, default="maps/challenge_p2.txt")
    ap.add_argument("--algo", type=str, default="astar", choices=["astar", "rbfs"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--max_expansions", type=int, default=None)
    ap.add_argument("--time_limit", type=float, default=None)
    ap.add_argument("--beam", type=int, default=None)

    ap.add_argument("--edge_lava", action="store_true")
    ap.add_argument("--no_edge_lava", action="store_true")

    ap.add_argument(
        "--respect_delay_penalty",
        action="store_true",
        help="If set, waiting time in the viewer will drain stamina (PDF delay penalty).",
    )

    ap.add_argument("--cell", type=int, default=52, help="Cell size in pixels")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    algo = args.algo
    edge_lava = (not args.no_edge_lava) if (args.edge_lava or args.no_edge_lava) else True

    max_depth = (14 if algo == "rbfs" else 28) if args.max_depth is None else int(args.max_depth)
    max_exp = (6000 if algo == "rbfs" else 12000) if args.max_expansions is None else int(args.max_expansions)
    time_limit = (6.0 if algo == "rbfs" else 12.0) if args.time_limit is None else float(args.time_limit)
    beam = (28 if algo == "rbfs" else 50) if args.beam is None else int(args.beam)

    import pygame

    pygame.init()
    font = pygame.font.SysFont("consolas", 18)

    env = ShoverWorldEnv(render_mode=None, map_path=args.map, edge_lava=edge_lava)
    obs = env.reset(seed=args.seed)

    info: Dict[str, Any] = {}
    if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
        obs, info = obs

    ai = PlayerAI(
        algorithm=algo,
        seed=args.seed,
        max_subgoal_depth=max_depth,
        max_expansions_per_subgoal=max_exp,
        planning_time_limit_sec=time_limit,
        edge_lava=edge_lava,
        beam_width=beam,
        prefer_hellify=True,
    )


    n_rows, n_cols = int(obs["grid"].shape[0]), int(obs["grid"].shape[1])
    cell = int(args.cell)
    hud_h = 80

    screen = pygame.display.set_mode((n_cols * cell, n_rows * cell + hud_h))
    pygame.display.set_caption(f"Watch AI – {algo}")

    clock = pygame.time.Clock()
    paused = False
    single_step = False
    steps_per_second = 5.0

    total_reward = 0.0
    step_count = 0

    last_action: Optional[Tuple[Tuple[int, int], int]] = None

    def reset_episode() -> None:
        nonlocal obs, info, ai, total_reward, step_count, last_action
        obs = env.reset(seed=args.seed)
        info = {}
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
            obs, info = obs
        ai = PlayerAI(
            algorithm=algo,
            seed=args.seed,
            max_subgoal_depth=max_depth,
            max_expansions_per_subgoal=max_exp,
            planning_time_limit_sec=time_limit,
            edge_lava=edge_lava,
            beam_width=beam,
            prefer_hellify=True,
        )
        total_reward = 0.0
        step_count = 0
        last_action = None

    def draw() -> None:
        screen.fill((18, 18, 22))
        grid = obs["grid"]

      
        for r in range(n_rows):
            for c in range(n_cols):
                v = int(grid[r, c])
                x = c * cell
                y = r * cell + hud_h

                pygame.draw.rect(screen, (30, 30, 36), (x, y, cell, cell))

                if v == LAVA:
                    pygame.draw.rect(screen, (140, 30, 30), (x + 4, y + 4, cell - 8, cell - 8))
                elif v == BARRIER:
                    pygame.draw.rect(screen, (110, 110, 120), (x + 3, y + 3, cell - 6, cell - 6))
                elif BOX_MIN <= v <= BOX_MAX:
                    pygame.draw.rect(screen, (160, 120, 60), (x + 5, y + 5, cell - 10, cell - 10))

         
                pygame.draw.rect(screen, (55, 55, 60), (x, y, cell, cell), 1)

        if last_action is not None:
            (pos, z) = last_action
            sr, sc = int(pos[0]), int(pos[1])
            if 0 <= sr < n_rows and 0 <= sc < n_cols and 0 <= int(z) <= 3:
                x = sc * cell
                y = sr * cell + hud_h
                pygame.draw.rect(screen, (240, 240, 80), (x + 2, y + 2, cell - 4, cell - 4), 3)

        stamina = float(np.array(info.get("stamina", obs["stamina"])) .reshape(-1)[0])
        boxes = int(info.get("number_of_boxes", np.logical_and(grid >= BOX_MIN, grid <= BOX_MAX).sum()))
        delay_pen = int(info.get("delay_penalty", 0))

        hud_lines = [
            f"algo={algo}   step={step_count}   total_reward={total_reward:.1f}",
            f"stamina={stamina:.1f}   boxes={boxes}   delay_pen={delay_pen}",
        ]
        if last_action is not None:
            (pos, z) = last_action
            z = int(z)
            if z == 4:
                a_str = "BARRIER_MAKER"
            elif z == 5:
                a_str = "HELLIFY"
            else:
                dir_str = ["UP", "RIGHT", "DOWN", "LEFT"][z]
                a_str = f"PUSH {dir_str} @ {tuple(pos)}"
            hud_lines.append(f"last_action: {a_str}")

        for i, line in enumerate(hud_lines):
            surf = font.render(line, True, (240, 240, 240))
            screen.blit(surf, (10, 10 + i * 22))

        if paused:
            surf = font.render("PAUSED (SPACE)", True, (255, 220, 120))
            screen.blit(surf, (n_cols * cell - 180, 10))

        pygame.display.flip()

    running = True
    last_step_wall = time.perf_counter()

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    single_step = False
                elif event.key == pygame.K_n:
                    if paused:
                        single_step = True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    steps_per_second = min(60.0, steps_per_second * 1.4)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    steps_per_second = max(0.5, steps_per_second / 1.4)
                elif event.key == pygame.K_r:
                    reset_episode()

        do_step = False
        if not paused:
            if (time.perf_counter() - last_step_wall) >= (1.0 / steps_per_second):
                do_step = True
        else:
            if single_step:
                do_step = True
                single_step = False

        if do_step:
            last_step_wall = time.perf_counter()
            if not args.respect_delay_penalty:
                env._last_step_walltime = time.perf_counter()

            action = ai.get_action(obs, info)
            last_action = action

            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            step_count += 1

            if done:
                paused = True

        draw()

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()