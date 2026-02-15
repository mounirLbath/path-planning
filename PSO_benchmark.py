import random
import time

from environment import load_problem
from PSO import particle_swarm_optimization, fitness


MAX_ITER = 1000
NB_POINTS_PATH = 10


def settings_line(scenario_number: int, setting: dict) -> str:
    """Single line listing all settings for this run."""
    parts = [f"Scenario {scenario_number}"]
    for k, v in setting.items():
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def settings():
    for S in (10, 30, 75, 100):
        for c1 in (1.2, 1.5, 2.0):
            for c2 in (1.2, 1.5, 2.0):
                for w in (0.7, 0.8, 0.9):
                    yield {
                        "S": S,
                        "max_iter": MAX_ITER,
                        "nb_points_path": NB_POINTS_PATH,
                        "c1": c1,
                        "c2": c2,
                        "w": w,
                        "random_restart": False,
                        "random_period": 15,
                        "simulated_annealing": False,
                        "beta": 0.95,
                        "T0": 10000,
                        "dim_learning": False,
                        "nb_update_before_dim": 15,
                    }


if __name__ == "__main__":
    # set random seed
    random.seed(42)

    for scenario_number in range(5):
        prob = load_problem(f"./scenarios/scenario{scenario_number}.txt")
        print("\n\nScenario ", scenario_number)
        for setting in settings():
            with open("PSO_benchmarks.txt") as f:
                lines = f.readlines()
                already_done = False
                settings_str_check = settings_line(scenario_number, setting)
                for line in lines:
                    if settings_str_check in line:
                        already_done = True
                        break
                if already_done:
                    print("Already have results for this setting, skipping.")
                    continue

            settings_str = settings_line(scenario_number, setting)
            print("\nSetting: ", setting)
            timer = time.time()
            best_path, _ = particle_swarm_optimization(
                prob,
                fitness=fitness,
                max_iter=setting["max_iter"],
                nb_points_path=setting["nb_points_path"],
                S=setting["S"],
                c1=setting["c1"],
                c2=setting["c2"],
                w=setting["w"],
                random_restart=setting["random_restart"],
                random_period=setting["random_period"],
                simulated_annealing=setting["simulated_annealing"],
                beta=setting["beta"],
                T0=setting["T0"],
                dim_learning=setting["dim_learning"],
                nb_update_before_dim=setting["nb_update_before_dim"],
                DEBUG=False,
                show_paths_start=False,
            )
            time_taken = time.time() - timer

            collisions = best_path.nb_pair_collision(prob.obstacles)
            cost = best_path.length()

            if collisions == 0:
                print(
                    f"Completed in {time_taken:.2f}s. Path total length {cost:.2f}"
                )
                with open("PSO_benchmarks.txt", "a") as f:
                    f.write(f"\n{settings_str}\n")
                    f.write(f"cost={cost:.2f}, time={time_taken:.2f}s\n")
            else:
                print(
                    f"PSO completed in {time_taken:.2f}s. No valid path found (collisions={collisions}, length={cost:.2f})."
                )
                with open("PSO_benchmarks.txt", "a") as f:
                    f.write(f"\n{settings_str}\n")
                    f.write(
                        f"No valid path found, collisions={collisions}, length={cost:.2f}, time={time_taken:.2f}s\n"
                    )
