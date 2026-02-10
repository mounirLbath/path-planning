import random
import time

from environment import load_problem
from RRT import rrt


def settings():
    for delta_s in (15, 30, 60):
        # 3
        for delta_r in (delta_s * 3, delta_s * 5):
            # 3*2 = 6
            for optimize_after_goal in (False, True):
                # 6 of each
                for max_iters in (1000, 2000) if optimize_after_goal else [1000]:
                    # 6 in 3 cases
                    yield {
                        "delta_s": delta_s,
                        "delta_r": delta_r,
                        "max_iters": max_iters,
                        "optimize_after_goal": optimize_after_goal,
                    }


if __name__ == "__main__":
    # set random seed
    random.seed(42)

    # Store results

    for scenario_number in range(5):
        prob = load_problem(f"./scenarios/scenario{scenario_number}.txt")
        print("\n\nScenario ", scenario_number)
        for setting in settings():
            with open("RRT_benchmarks.txt") as f:
                lines = f.readlines()
                already_done = False
                for i, line in enumerate(lines):
                    if (
                        f"Scenario {scenario_number}, delta_s={setting['delta_s']}, delta_r={setting['delta_r']}"
                        in line
                        and (
                            (f"optimized_for={setting['max_iters']}" in lines[i + 1] and setting["optimize_after_goal"])
                            or ("steps_taken" in lines[i + 1] and not setting["optimize_after_goal"])
                        )
                    ):
                        already_done = True
                if already_done:
                    print("Already have results for this setting, skipping.")
                    continue

            # 18 settings * 5 scenarios = 90 runs total
            print("\nSetting: ", setting)
            timer = time.time()
            cost, path, steps_taken, _ = rrt(
                prob,
                delta_s=setting["delta_s"],
                delta_r=setting["delta_r"],
                max_iters=setting["max_iters"],
                recursive_rewire=False,
                optimize_after_goal=setting["optimize_after_goal"],
                display_tree_end=False,
            )
            time_taken = time.time() - timer
            if path is not None:
                print(
                    f"Completed in {time_taken:.2f}s. Path total length {cost:.2f}"
                    + (
                        f", found in {steps_taken} steps"
                        if not setting["optimize_after_goal"]
                        else f", optimized over {setting['max_iters']} steps."
                    )
                )
                with open("RRT_benchmarks.txt", "a") as f:
                    f.write(
                        f"\nScenario {scenario_number}, delta_s={setting['delta_s']}, delta_r={setting['delta_r']}\ncost={cost:.2f}, time={time_taken:.2f}s, "
                        + (
                            f"steps_taken={steps_taken}\n"
                            if not setting["optimize_after_goal"]
                            else f"optimized_for={setting['max_iters']}\n"
                        )
                    )

            else:
                print(f"RRT completed in {time_taken:.2f} seconds. No path found.")
                with open("RRT_benchmarks.txt", "a") as f:
                    f.write(
                        f"\nScenario {scenario_number}, delta_s={setting['delta_s']}, delta_r={setting['delta_r']}\nNo path found in steps_taken={steps_taken}, time={time_taken:.2f}s,"
                    )
