from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

VARIANTS = ("minimax", "alpha_beta")
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


# --- Klasa opisująca zasady gry ---
class SaveResult:
    def __init__(
        self,
        variant: str,
        depth: int,
        games: int,
        win_rate_pct: float,
        avg_time_ms: float,
        avg_nodes: float,
    ) -> None:
        self.variant = variant
        self.depth = depth
        self.games = games
        self.win_rate_pct = win_rate_pct
        self.avg_time_ms = avg_time_ms
        self.avg_nodes = avg_nodes

    @staticmethod
    def format_results_table(results: list[SaveResult]) -> str:
        """Generuje sformatowaną tabelę Markdown z wynikami."""
        header = (
            "| Wariant    | Głębokość d | Partie | Wygrane [%] "
            "| Średni czas ruchu [ms] | Średnia liczba węzłów |"
        )
        separator = (
            "|:-----------|------------:|-------:|------------:"
            "|-----------------------:|----------------------:|"
        )
        rows = [header, separator]
        for row in results:
            rows.append(
                f"| {row.variant:<10} | "
                f"{row.depth:>11} | "
                f"{row.games:>6} | "
                f"{row.win_rate_pct:>11.1f} | "
                f"{row.avg_time_ms:>22.3f} | "
                f"{row.avg_nodes:>21.2f} |"
            )
        return "\n".join(rows)

    @staticmethod
    def save_results(results: list[SaveResult]) -> None:
        csv_path = Path("results.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "variant",
                    "depth",
                    "games",
                    "win_rate_pct",
                    "avg_time_ms",
                    "avg_nodes",
                ],
            )
            writer.writeheader()
            writer.writerows([row.__dict__ for row in results])

        report_path = Path("results.md")
        report_path.write_text(
            SaveResult.format_results_table(results) + "\n", encoding="utf-8"
        )


# --- Klasa zarządzająca zasadami gry ---
class NimRules:
    """Przechowuje stan i zasady konkretnego wariantu gry."""

    def __init__(self, max_take: int, max_tokens: int):
        self.max_take = max_take
        self.max_tokens = max_tokens

    def legal_moves(self, tokens_left: int) -> list[int]:
        return list(range(1, min(tokens_left, self.max_take) + 1))

    def apply_move(self, state: GameState, move: int) -> GameState:
        return GameState(
            tokens_left=state.tokens_left - move,
            current_player=1 - state.current_player,
        )

    def evaluate_state(self, state: GameState, maximizing_player: int) -> float:

        if state.tokens_left == 0:
            return 1.0 if state.current_player == maximizing_player else -1.0

        # Wariant misere
        losing_for_player_to_move = (
            state.tokens_left % (self.max_take + 1) == 1
        )
        strategic_score = 0.75 if losing_for_player_to_move else -0.75

        if state.current_player == maximizing_player:
            strategic_score *= -1

        progress_bonus = 0.25 * (1.0 - state.tokens_left / self.max_tokens)
        if state.current_player != maximizing_player:
            progress_bonus *= -1

        return strategic_score + progress_bonus


# --- Klasy Agentów (Graczy) ---
class MinimaxAgent:
    """Agent wykorzystujący algorytm Minimax (opcjonalnie z Alpha-Beta)."""

    def __init__(
        self, rules: NimRules, depth: int, variant: str, rng: random.Random
    ):
        self.rules = rules
        self.depth = depth
        self.use_alpha_beta = variant == "alpha_beta"
        self.rng = rng

    def _minimax_value(
        self,
        state: GameState,
        depth: int,
        maximizing_player: int,
        stats: SearchStats,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> float:
        stats.visited_nodes += 1

        if depth == 0 or state.tokens_left == 0:
            return self.rules.evaluate_state(state, maximizing_player)

        moves = self.rules.legal_moves(state.tokens_left)

        if state.current_player == maximizing_player:
            is_max_turn = True
        else:
            is_max_turn = False

        if is_max_turn:
            best_value = float("-inf")
            for move in moves:
                child = self.rules.apply_move(state, move)
                value = self._minimax_value(
                    child, depth - 1, maximizing_player, stats, alpha, beta
                )
                best_value = max(best_value, value)
                if self.use_alpha_beta:
                    alpha = max(alpha, best_value)
                    if beta <= alpha:
                        break
            return best_value

        best_value = float("inf")
        for move in moves:
            child = self.rules.apply_move(state, move)
            value = self._minimax_value(
                child, depth - 1, maximizing_player, stats, alpha, beta
            )
            best_value = min(best_value, value)
            if self.use_alpha_beta:
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
        return best_value

    def choose_move(self, state: GameState) -> MoveDecision:
        stats = SearchStats()
        start = time.perf_counter()
        moves = self.rules.legal_moves(state.tokens_left)
        maximizing_player = state.current_player

        scored_moves: list[tuple[int, float]] = []

        for move in moves:
            child = self.rules.apply_move(state, move)
            value = self._minimax_value(
                child,
                self.depth - 1,
                maximizing_player,
                stats,
                float("-inf"),
                float("inf"),
            )
            scored_moves.append((move, value))

        best_score = max(score for _, score in scored_moves)
        best_moves = [
            move for move, score in scored_moves if score == best_score
        ]
        selected_move = self.rng.choice(best_moves)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return MoveDecision(
            move=selected_move,
            score=best_score,
            visited_nodes=stats.visited_nodes,
            elapsed_ms=elapsed_ms,
        )


class RandomAgent:
    """Agent wykonujący losowe, ale legalne ruchy."""

    def __init__(self, rules: NimRules, rng: random.Random):
        self.rules = rules
        self.rng = rng

    def choose_move(self, state: GameState) -> int:
        return self.rng.choice(self.rules.legal_moves(state.tokens_left))


# --- Klasa Symulatora Gry ---
class NimSimulator:
    """Odpowiada za pętlę gry i zbieranie statystyk."""

    def __init__(
        self, rules: NimRules, agent_p1: MinimaxAgent, agent_p2: RandomAgent
    ):
        self.rules = rules
        self.agent_p1 = agent_p1  # Gracz 0
        self.agent_p2 = agent_p2  # Gracz 1

    def play_game(self, initial_tokens: int) -> dict[str, float]:
        state = GameState(tokens_left=initial_tokens, current_player=0)
        total_nodes = 0
        total_time_ms = 0.0
        agent_moves = 0

        while True:
            if state.current_player == 0:
                decision = self.agent_p1.choose_move(state)
                move = decision.move
                total_nodes += decision.visited_nodes
                total_time_ms += decision.elapsed_ms
                agent_moves += 1
            else:
                move = self.agent_p2.choose_move(state)

            next_state = self.rules.apply_move(state, move)

            if next_state.tokens_left == 0:
                winner = next_state.current_player
                return {
                    "won": 1.0 if winner == 0 else 0.0,
                    "avg_nodes": (
                        total_nodes / agent_moves if agent_moves else 0.0
                    ),
                    "avg_time_ms": (
                        total_time_ms / agent_moves if agent_moves else 0.0
                    ),
                }

            state = next_state


def run_experiments() -> list[SaveResult]:
    """Przeprowadza symulacje. Zwraca listę obiektów SaveResult z wynikami."""
    results = []

    # Inicjalizacja losowania ilości początkowych żetonów
    shared_rng = random.Random(BASE_SEED)
    sampled_tokens = shared_rng.choices(
        range(MIN_TOKENS, MAX_TOKENS + 1), k=GAMES_PER_DEPTH
    )

    # Inicjalizacja zasad gry
    rules = NimRules(max_take=MAX_TAKE, max_tokens=MAX_TOKENS)

    for variant in VARIANTS:
        for depth in DEPTHS:
            search_rng = random.Random(BASE_SEED + depth)
            opponent_rng = random.Random(BASE_SEED + 99)

            agent_p1 = MinimaxAgent(rules, depth, variant, search_rng)
            agent_p2 = RandomAgent(rules, opponent_rng)
            simulator = NimSimulator(rules, agent_p1, agent_p2)

            total_wins = 0.0
            total_time_ms = 0.0
            total_nodes = 0.0

            for tokens in sampled_tokens:
                game_stats = simulator.play_game(initial_tokens=tokens)
                total_wins += game_stats["won"]
                total_time_ms += game_stats["avg_time_ms"]
                total_nodes += game_stats["avg_nodes"]

            # Utworzenie i dodanie obiektu SaveResult
            results.append(
                SaveResult(
                    variant=variant,
                    depth=depth,
                    games=GAMES_PER_DEPTH,
                    win_rate_pct=(total_wins / GAMES_PER_DEPTH) * 100,
                    avg_time_ms=total_time_ms / GAMES_PER_DEPTH,
                    avg_nodes=total_nodes / GAMES_PER_DEPTH,
                )
            )

    return results


def main() -> None:
    results = run_experiments()
    SaveResult.save_results(results)
    print(SaveResult.format_results_table(results))


if __name__ == "__main__":
    main()
