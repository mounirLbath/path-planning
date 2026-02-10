def parse_line_scenario(line):
    scenario_number = int(line.split()[1][0])
    delta_s = int(line.split("delta_s=")[1].split(",")[0])
    delta_r = int(line.split("delta_r=")[1].rstrip())
    return scenario_number, delta_s, delta_r


def parse_line_stats(line):
    cost = float(line.split("cost=")[1].split(",")[0])
    time_taken = float(line.split("time=")[1].split("s")[0])
    if "steps_taken=" in line:
        optimized = False
        steps_taken = int(line.split("steps_taken=")[1].rstrip())
    else:
        optimized = True
        steps_taken = int(line.split("optimized_for=")[1].rstrip())
    return cost, time_taken, optimized, steps_taken


if __name__ == "__main__":
    with open("RRT_benchmarks.txt") as f:
        lines = f.readlines()

    scenario_best = [dict() for _ in range(5)]
    for line in lines:
        if line.startswith("Scenario"):
            scenario_number, delta_s, delta_r = parse_line_scenario(line)
        elif line.startswith("cost="):
            cost, time_taken, optimized, steps_taken = parse_line_stats(line)

            stats = {
                "delta_s": delta_s,
                "delta_r": delta_r,
                "steps_taken": steps_taken,
                "optimized": optimized,
                "cost": cost,
                "time": time_taken,
                "value": 1,  # If it is best_cost or best_time, its value is 1 by definition
            }

            if cost < scenario_best[scenario_number].get("best_cost", {"cost": float("inf")})["cost"]:
                scenario_best[scenario_number]["best_cost"] = stats
            if time_taken < scenario_best[scenario_number].get("best_time", {"time": float("inf")})["time"]:
                scenario_best[scenario_number]["best_time"] = stats

    for line in lines:
        if line.startswith("Scenario"):
            scenario_number, delta_s, delta_r = parse_line_scenario(line)
        elif line.startswith("cost="):
            cost, time_taken, optimized, steps_taken = parse_line_stats(line)

            best_cost = scenario_best[scenario_number]["best_cost"]["cost"]
            slow_time = scenario_best[scenario_number]["best_cost"]["time"]

            best_time = scenario_best[scenario_number]["best_time"]["time"]
            large_cost = scenario_best[scenario_number]["best_time"]["cost"]

            assert best_cost <= large_cost
            assert best_time <= slow_time

            # value = %close to best cost / %close to worst time
            value = (1 - ((cost - best_cost) / (large_cost - best_cost + 1e-6))) / (
                1 - ((time_taken - slow_time) / (best_time - slow_time + 1e-6))
            )

            if value > scenario_best[scenario_number].get("best_value", {"value": 0})["value"]:
                scenario_best[scenario_number]["best_value"] = stats
                stats = {
                    "delta_s": delta_s,
                    "delta_r": delta_r,
                    "steps_taken": steps_taken,
                    "optimized": optimized,
                    "cost": cost,
                    "time": time_taken,
                    "value": value,
                }

    with open("RRT_best_performances.txt", "w") as f:
        for i, best in enumerate(scenario_best):
            f.write(f"\nScenario {i}:\n")
            for metric in ["best_value", "best_cost", "best_time"]:
                stats = best[metric]
                f.write(
                    f"  {metric}: delta_s={stats['delta_s']}, delta_r={stats['delta_r']}, steps_taken={stats['steps_taken']}, optimized={stats['optimized']}, cost={stats['cost']:.2f}, time={stats['time']:.2f}s, value={stats['value']:.2f}\n\n"
                )
