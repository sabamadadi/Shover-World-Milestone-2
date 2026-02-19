from __future__ import annotations
import argparse
import time
from typing import Any, Dict
from environment import ShoverWorldEnv
from player_ai import PlayerAI

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=str, default="C:\Users\Asus\Desktop\maps\maps\map3.txt", help="Path to map file")
    ap.add_argument("--algo", type=str, default="astar", choices=["astar", "rbfs"], help="Search algorithm")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max_depth", type=int, default=None, help="Maximum search depth per sub-goal")
    ap.add_argument("--max_expansions", type=int, default=None, help="Maximum node expansions per sub-goal")
    ap.add_argument("--time_limit", type=float, default=None, help="Time limit (seconds) per sub-goal")
    ap.add_argument("--beam", type=int, default=None, help="Beam width (top-K successors per expansion)")

    ap.add_argument("--max_steps", type=int, default=400)

    ap.add_argument("--edge_lava", action="store_true", help="Treat pushing off-board as lava/removal (recommended)")
    ap.add_argument("--no_edge_lava", action="store_true", help="Disable edge-lava behavior")

    ap.add_argument("--quiet", action="store_true", help="Less per-step logging")
    args = ap.parse_args()


    edge_lava = (not args.no_edge_lava) if (args.edge_lava or args.no_edge_lava) else True

  
    if args.algo == "rbfs":
        max_depth = 12 if args.max_depth is None else int(args.max_depth)
        max_exp = 6000 if args.max_expansions is None else int(args.max_expansions)
        tlim = 5.0 if args.time_limit is None else float(args.time_limit)
        beam = 25 if args.beam is None else int(args.beam)
    else:
        max_depth = 30 if args.max_depth is None else int(args.max_depth)
        max_exp = 20000 if args.max_expansions is None else int(args.max_expansions)
        tlim = 20.0 if args.time_limit is None else float(args.time_limit)
        beam = 60 if args.beam is None else int(args.beam)

    env = ShoverWorldEnv(render_mode=None, map_path=args.map, edge_lava=edge_lava)
    obs = env.reset(seed=args.seed)

    info: Dict[str, Any] = {}
    if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
        obs, info = obs

    ai = PlayerAI(
        algorithm=args.algo,
        seed=args.seed,
        edge_lava=edge_lava,
        max_subgoal_depth=max_depth,
        max_expansions_per_subgoal=max_exp,
        time_limit_sec=tlim,
        beam_width=beam,
 
        prefer_hellify=True,
    )

    total_reward = 0.0
    step = 0

    t0 = time.perf_counter()

    while step < args.max_steps:
        action = ai.get_action(obs, info)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        step += 1

        if not args.quiet:
            print(
                f"Step {step:3d} | reward {reward:7.2f} | stamina {info.get('stamina', 0):8.2f} "
                f"| boxes {info.get('number_of_boxes', '?')} | delay_pen {info.get('delay_penalty', 0)} "
                f"| action {action}"
            )

        if done:
            break

    dt = time.perf_counter() - t0

    print("\n=== Finished ===")
    print(f"Algorithm:    {args.algo}")
    print(f"Steps:        {step}")
    print(f"Runtime (s):  {dt:.3f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Boxes left:   {info.get('number_of_boxes', None)}")
    print(f"Stamina:      {info.get('stamina', None)}")

    env.close()


if __name__ == "__main__":
    main()