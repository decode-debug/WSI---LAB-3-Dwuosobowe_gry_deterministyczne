from game_logic import run_experiments
from reporting import format_results_table, save_results


def main() -> None:
    results = run_experiments()
    save_results(results)
    print(format_results_table(results))


if __name__ == "__main__":
    main()
