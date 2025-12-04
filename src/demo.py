import argparse
from typing import List

import jax
from jax import random

from .simulation import (
    SCHCParams,
    SCHCState,
    advance_one_step,
    initialize_state,
)


def _render(config, outfile: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.imshow(config, cmap="hsv", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a Structural Cellular Hash Chemistry demo.")
    parser.add_argument("--steps", type=int, default=200, help="Number of onestep iterations to run.")
    parser.add_argument("--size", type=int, default=100, help="Grid size L.")
    parser.add_argument("--k", type=int, default=1000, help="Number of possible cell types.")
    parser.add_argument("--n", type=int, default=10, help="Initial number of active cells.")
    parser.add_argument("--mu", type=float, default=0.002 / 0.999, help="Mutation probability for non-empty cells.")
    parser.add_argument("--death-prob", type=float, default=0.001, help="Probability a copied cell vanishes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--outfile", type=str, default="", help="If set, save final grid as an image.")
    args = parser.parse_args(argv)

    params = SCHCParams(
        k=args.k,
        n=args.n,
        L=args.size,
        mu=args.mu,
        death_prob=args.death_prob,
    )
    key = random.PRNGKey(args.seed)
    state: SCHCState = initialize_state(key, params)

    for step in range(args.steps):
        state = advance_one_step(state, params)
        if (step + 1) % max(1, args.steps // 10) == 0:
            num_active = int((state.config > 0).sum())
            print(f"t={state.time:04d} active={num_active}")

    if args.outfile:
        _render(state.config, args.outfile)
        print(f"Final grid saved to {args.outfile}")


if __name__ == "__main__":
    main()
