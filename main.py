from __future__ import annotations

import csv
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


MIN_TOKENS = 8
MAX_TOKENS = 20
MAX_TAKE = 3
DEPTHS = (2, 3, 4, 5)
GAMES_PER_DEPTH = 200
BASE_SEED = 20260419


@dataclass(frozen=True)
class GameState:
    tokens_left: int
    current_player: int


@dataclass
class SearchStats:
    visited_nodes: int = 0


@dataclass
class MoveDecision:
    move: int
    score: float
    visited_nodes: int
    elapsed_ms: float


def legal_moves(tokens_left: int, max_take: int = MAX_TAKE) -> list[int]:
    return list(range(1, min(tokens_left, max_take) + 1))


def apply_move(state: GameState, move: int) -> GameState:
    return GameState(tokens_left=state.tokens_left - move, current_player=1 - state.current_player)


def evaluate_state(state: GameState, maximizing_player: int) -> float:
    if state.tokens_left == 0:
        return 1.0 if state.current_player == maximizing_player else -1.0

    # In this misere variant the positions n = 1 (mod MAX_TAKE + 1) are losing
    # for the player who is about to move under perfect play.
    losing_for_player_to_move = state.tokens_left % (MAX_TAKE + 1) == 1
    strategic_score = 0.75 if losing_for_player_to_move else -0.75
    if state.current_player == maximizing_player:
        strategic_score *= -1

    # Prefer positions that are closer to a forced result.
    progress_bonus = 0.25 * (1.0 - state.tokens_left / MAX_TOKENS)
    if state.current_player != maximizing_player:
        progress_bonus *= -1

    return strategic_score + progress_bonus


def minimax_value(
    state: GameState,
    depth: int,
    maximizing_player: int,
    stats: SearchStats,
    use_alpha_beta: bool,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
) -> float:
    stats.visited_nodes += 1

    if depth == 0 or state.tokens_left == 0:
        return evaluate_state(state, maximizing_player)

    moves = legal_moves(state.tokens_left)
    is_max_turn = state.current_player == maximizing_player

    if is_max_turn:
        best_value = float("-inf")
        for move in moves:
            child = apply_move(state, move)
            value = minimax_value(child, depth - 1, maximizing_player, stats, use_alpha_beta, alpha, beta)
            best_value = max(best_value, value)
            if use_alpha_beta:
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
        return best_value

    best_value = float("inf")
    for move in moves:
        child = apply_move(state, move)
        value = minimax_value(child, depth - 1, maximizing_player, stats, use_alpha_beta, alpha, beta)
        best_value = min(best_value, value)
        if use_alpha_beta:
            beta = min(beta, best_value)
            if beta <= alpha:
                break
    return best_value


def choose_move(
    state: GameState,
    depth: int,
    rng: random.Random,
    variant: str,
) -> MoveDecision:
    stats = SearchStats()
    start = time.perf_counter()
    moves = legal_moves(state.tokens_left)
    maximizing_player = state.current_player
    use_alpha_beta = variant == "alpha_beta"

    scored_moves: list[tuple[int, float]] = []

    for move in moves:
        child = apply_move(state, move)
        value = minimax_value(
            child,
            depth - 1,
            maximizing_player,
            stats,
            use_alpha_beta,
            float("-inf"),
            float("inf"),
        )
        scored_moves.append((move, value))

    best_score = max(score for _, score in scored_moves)
    best_moves = [move for move, score in scored_moves if score == best_score]
    selected_move = rng.choice(best_moves)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return MoveDecision(
        move=selected_move,
        score=best_score,
        visited_nodes=stats.visited_nodes,
        elapsed_ms=elapsed_ms,
    )


def random_move(state: GameState, rng: random.Random) -> int:
    return rng.choice(legal_moves(state.tokens_left))


def play_game(
    initial_tokens: int,
    depth: int,
    variant: str,
    search_rng: random.Random,
    opponent_rng: random.Random,
) -> dict[str, float]:
    state = GameState(tokens_left=initial_tokens, current_player=0)
    total_nodes = 0
    total_time_ms = 0.0
    agent_moves = 0

    while True:
        if state.current_player == 0:
            decision = choose_move(state, depth, search_rng, variant)
            move = decision.move
            total_nodes += decision.visited_nodes
            total_time_ms += decision.elapsed_ms
            agent_moves += 1
        else:
            move = random_move(state, opponent_rng)

        next_state = apply_move(state, move)

        if next_state.tokens_left == 0:
            winner = next_state.current_player
            return {
                "won": 1.0 if winner == 0 else 0.0,
                "avg_nodes": total_nodes / agent_moves if agent_moves else 0.0,
                "avg_time_ms": total_time_ms / agent_moves if agent_moves else 0.0,
            }

        state = next_state


def run_experiments() -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    shared_rng = random.Random(BASE_SEED)
    sampled_tokens = [shared_rng.randint(MIN_TOKENS, MAX_TOKENS) for _ in range(GAMES_PER_DEPTH)]

    for variant in ("minimax", "alpha_beta"):
        for depth in DEPTHS:
            wins: list[float] = []
            avg_nodes: list[float] = []
            avg_times: list[float] = []

            for game_index, initial_tokens in enumerate(sampled_tokens):
                search_rng = random.Random(BASE_SEED + depth * 1000 + game_index)
                opponent_rng = random.Random(BASE_SEED + 50_000 + depth * 1000 + game_index)
                outcome = play_game(initial_tokens, depth, variant, search_rng, opponent_rng)
                wins.append(outcome["won"])
                avg_nodes.append(outcome["avg_nodes"])
                avg_times.append(outcome["avg_time_ms"])

            results.append(
                {
                    "variant": variant,
                    "depth": depth,
                    "games": GAMES_PER_DEPTH,
                    "win_rate_pct": 100.0 * statistics.fmean(wins),
                    "avg_time_ms": statistics.fmean(avg_times),
                    "avg_nodes": statistics.fmean(avg_nodes),
                }
            )

    return results


def format_results_table(results: list[dict[str, float | int | str]]) -> str:
    header = "| Wariant | Glebokosc d | Partie | Wygrane [%] | Sredni czas ruchu [ms] | Srednia liczba wezlow |"
    separator = "|---|---:|---:|---:|---:|---:|"
    rows = [header, separator]

    for row in results:
        rows.append(
            "| {variant} | {depth} | {games} | {win_rate_pct:.1f} | {avg_time_ms:.3f} | {avg_nodes:.2f} |".format(
                **row
            )
        )

    return "\n".join(rows)


def save_results(results: list[dict[str, float | int | str]]) -> None:
    csv_path = Path("results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["variant", "depth", "games", "win_rate_pct", "avg_time_ms", "avg_nodes"],
        )
        writer.writeheader()
        writer.writerows(results)

    report_path = Path("results.md")
    report_path.write_text(format_results_table(results) + "\n", encoding="utf-8")


def main() -> None:
    results = run_experiments()
    save_results(results)
    print(format_results_table(results))


if __name__ == "__main__":
    main()
