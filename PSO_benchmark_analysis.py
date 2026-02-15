"""Analyze PSO_benchmarks.txt and report best cost, best time, and best value per scenario."""

import re
from typing import Any


def parse_value(s: str):
    s = s.strip()
    if s == "True":
        return True
    if s == "False":
        return False
    try:
        return int(s)
    except:
        pass
    try:
        return float(s)
    except:
        return s


def parse_settings_line(line: str) -> tuple[int, dict]:
    line = line.strip()
    if not line.startswith("Scenario"):
        return -1, {}
    parts = [p.strip() for p in line.split(",")]
    scenario_str = parts[0]  # "Scenario 0"
    scenario_number = int(scenario_str.split()[1])
    settings = {}
    for p in parts[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        settings[k.strip()] = parse_value(v)
    return scenario_number, settings


def parse_result_line(line: str) -> tuple[float | None, float | None]:
    """
    Parse a result line. Returns (cost, time) if valid path, else (None, time) for no path.
    """
    line = line.strip()
    time_match = re.search(r"time=([\d.]+)s", line)
    time_taken = float(time_match.group(1)) if time_match else None
    if line.startswith("cost="):
        cost_match = re.search(r"cost=([\d.]+)", line)
        cost = float(cost_match.group(1)) if cost_match else None
        return cost, time_taken
    # "No valid path found, collisions=..., length=..., time=...s"
    return None, time_taken


def format_settings(settings: dict) -> str:
    """Format settings dict as a short string for output."""
    return ", ".join(f"{k}={v}" for k, v in settings.items())


if __name__ == "__main__":
    with open("PSO_benchmarks.txt") as f:
        lines = f.readlines()

    # Index by scenario; store best_cost, best_time, best_value (with full stats)
    scenario_best = [dict() for _ in range(5)]

    i = 0
    while i < len(lines):
        line = lines[i]
        scenario_number, settings = parse_settings_line(line)
        if scenario_number < 0:
            i += 1
            continue
        i += 1
        if i >= len(lines):
            break
        cost, time_taken = parse_result_line(lines[i])
        i += 1

        if time_taken is None:
            continue

        if cost is not None:
            stats = {
                "settings": settings,
                "cost": cost,
                "time": time_taken,
                "value": 1.0,
            }

            if cost < scenario_best[scenario_number].get("best_cost", {"cost": float("inf")})["cost"]:
                scenario_best[scenario_number]["best_cost"] = stats.copy()
            if time_taken < scenario_best[scenario_number].get("best_time", {"time": float("inf")})["time"]:
                scenario_best[scenario_number]["best_time"] = stats.copy()

    # Second pass: compute value for every run that found a path, then pick best_value
    i = 0
    while i < len(lines):
        line = lines[i]
        scenario_number, settings = parse_settings_line(line)
        if scenario_number < 0:
            i += 1
            continue
        i += 1
        if i >= len(lines):
            break
        cost, time_taken = parse_result_line(lines[i])
        i += 1

        if cost is None or time_taken is None:
            continue

        best = scenario_best[scenario_number]
        best_cost_stats = best.get("best_cost")
        best_time_stats = best.get("best_time")
        if not best_cost_stats or not best_time_stats:
            continue

        best_cost = best_cost_stats["cost"]
        slow_time = best_cost_stats["time"]
        best_time = best_time_stats["time"]
        large_cost = best_time_stats["cost"]

    
        assert best_cost <= large_cost
        assert best_time <= slow_time

        denom_cost = large_cost - best_cost + 1e-6
        denom_time = best_time - slow_time + 1e-6
        time_ratio = (time_taken - slow_time) / denom_time
        value = (1 - (cost - best_cost) / denom_cost) / (1 - time_ratio + 1e-6)

        stats = {
            "settings": settings,
            "cost": cost,
            "time": time_taken,
            "value": value,
        }
        if value > best.get("best_value", {"value": -float("inf")})["value"]:
            scenario_best[scenario_number]["best_value"] = stats

    with open("PSO_best_performances.txt", "w") as f:
        for idx, best in enumerate(scenario_best):
            f.write(f"\nScenario {idx}:\n")
            for metric in ["best_value", "best_cost", "best_time"]:
                if metric not in best:
                    f.write(f"  {metric}: (no valid runs)\n\n")
                    continue
                s = best[metric]
                f.write(
                    f"  {metric}: {format_settings(s['settings'])}\n"
                    f"    cost={s['cost']:.2f}, time={s['time']:.2f}s, value={s['value']:.2f}\n\n"
                )

    print("Wrote PSO_best_performances.txt")
