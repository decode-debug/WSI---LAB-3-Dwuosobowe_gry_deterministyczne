from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """All tuneable parameters for a run."""

    variants: tuple[str, ...] = field(
        default_factory=lambda: ("minimax", "alpha_beta")
    )
    min_tokens: int = 8
    max_tokens: int = 20
    max_take: int = 3
    depths: tuple[int, ...] = field(default_factory=lambda: (2, 3, 4, 5))
    games_per_depth: int = 200
    base_seed: int = 20260419


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


@dataclass
class ExperimentResult:
    variant: str
    depth: int
    games: int
    win_rate_pct: float
    avg_time_ms: float
    avg_nodes: float


class NimRules:
    """Stores game constraints and state transition helpers."""

    def __init__(self, max_take: int, max_tokens: int):
        self.max_take = max_take
        self.max_tokens = max_tokens

    def legal_moves(self, tokens_left: int) -> list[int]:
        """Returns a list of legal moves (number of tokens to take)."""
        return list(range(1, min(tokens_left, self.max_take) + 1))

    def apply_move(self, state: GameState, move: int) -> GameState:
        """Returns the new game state after applying the move."""
        return GameState(
            tokens_left=state.tokens_left - move,
            current_player=1 - state.current_player,
        )

    def evaluate_state(
        self,
        state: GameState,
        maximizing_player: int,
    ) -> float:
        """Heuristic evaluation of the game state from the perspective of maximizing_player."""

        # stan końcowy: jeśli nie ma tokenów,
        # gracz, który właśnie wykonał ruch, wygrał
        if state.tokens_left == 0:
            return 1.0 if state.current_player == maximizing_player else -1.0

        # strategia: gracz mający ruch jest w pozycji przegranej (niekorzystnej),
        # jeśli liczba tokenów pozostających do wzięcia jest równa 1 modulo (max_take + 1)
        losing_for_player_to_move = (
            state.tokens_left % (self.max_take + 1) == 1
        )
        strategic_score = 0.75 if losing_for_player_to_move else -0.75

        # jeśli obecny gracz jest maksymalizowanym, odwracamy ocenę strategiczną,
        if state.current_player == maximizing_player:
            strategic_score *= -1

        # bonus za postęp w grze, który rośnie w miarę zmniejszania się liczby tokenów,
        progress_bonus = 0.25 * (1.0 - state.tokens_left / self.max_tokens)

        # jeśli obecny gracz nie jest maksymalizowanym, odwracamy bonus postępu,
        if state.current_player != maximizing_player:
            progress_bonus *= -1

        return strategic_score + progress_bonus


class MinimaxAgent:
    """Minimax player with optional alpha-beta pruning."""

    def __init__(
        self,
        rules: NimRules,
        depth: int,
        variant: str,
        rng: random.Random,
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
        """Returns the minimax value of the state."""

        # jeśli osiągnięto maksymalną głębokość lub stan końcowy, zwróć ocenę heurystyczną
        if depth == 0 or state.tokens_left == 0:
            return self.rules.evaluate_state(state, maximizing_player)

        moves = self.rules.legal_moves(state.tokens_left)
        is_max_turn = state.current_player == maximizing_player

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
    """Opponent making random legal moves."""

    def __init__(self, rules: NimRules, rng: random.Random):
        self.rules = rules
        self.rng = rng

    def choose_move(self, state: GameState) -> int:
        # wybiera losowy ruch spośród legalnych
        return self.rng.choice(self.rules.legal_moves(state.tokens_left))


class NimSimulator:
    """Runs one game and collects per-move agent statistics."""

    def __init__(
        self,
        rules: NimRules,
        agent_p1: MinimaxAgent,
        agent_p2: RandomAgent,
    ):
        self.rules = rules
        self.agent_p1 = agent_p1  # player 0
        self.agent_p2 = agent_p2  # player 1

    def play_game(self, initial_tokens: int) -> dict[str, float]:
        """Plays a single game and returns statistics for the Minimax agent."""
        state = GameState(tokens_left=initial_tokens, current_player=0)
        total_nodes = 0
        total_time_ms = 0.0
        agent_moves = 0

        # gra toczy się do momentu, aż nie będzie tokenów
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


def run_experiments(
    cfg: ExperimentConfig | None = None,
) -> list[ExperimentResult]:
    """Runs all simulations and returns aggregated metrics."""

    if cfg is None:
        cfg = ExperimentConfig()

    results: list[ExperimentResult] = []

    # generuje losowe liczby tokenów dla każdej gry,
    # aby zapewnić różnorodność scenariuszy,
    # ale z powtarzalnością dzięki bazowemu seedowi
    sampled_tokens = random.Random(cfg.base_seed).choices(
        range(cfg.min_tokens, cfg.max_tokens + 1),
        k=cfg.games_per_depth,
    )

    # inicjalizuje zasady gry
    rules = NimRules(max_take=cfg.max_take, max_tokens=cfg.max_tokens)

    # iteruje przez wszystkie kombinacje
    # wariantów ("minimax", "alpha_beta") i głębokościach (2, 3, 4, 5)
    for variant in cfg.variants:
        for depth in cfg.depths:
            # tworzy osobne generatory liczb losowych dla agenta i przeciwnika,
            search_rng = random.Random(cfg.base_seed + depth)
            opponent_rng = random.Random(cfg.base_seed + 99)

            # tworzy agenta Minimax i losowego przeciwnika, oraz symulator gry
            agent_p1 = MinimaxAgent(rules, depth, variant, search_rng)
            agent_p2 = RandomAgent(rules, opponent_rng)
            simulator = NimSimulator(rules, agent_p1, agent_p2)

            # resetuje liczniki wyników dla kombinacji wariantu i głębokości
            total_wins = 0.0
            total_time_ms = 0.0
            total_nodes = 0.0

            # rozgrywa określoną liczbę gier dla tej konfiguracji,
            # używając wcześniej wygenerowanych losowych tokenów startowych,
            for tokens in sampled_tokens:
                game_stats = simulator.play_game(initial_tokens=tokens)
                total_wins += game_stats["won"]
                total_time_ms += game_stats["avg_time_ms"]
                total_nodes += game_stats["avg_nodes"]

            # po rozegraniu wszystkich gier, oblicza średnie i zapisuje wyniki
            results.append(
                ExperimentResult(
                    variant=variant,
                    depth=depth,
                    games=cfg.games_per_depth,
                    win_rate_pct=(total_wins / cfg.games_per_depth) * 100,
                    avg_time_ms=total_time_ms / cfg.games_per_depth,
                    avg_nodes=total_nodes / cfg.games_per_depth,
                )
            )

    return results
