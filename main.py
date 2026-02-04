
from environment import *

if __name__ == "__main__":
    problem = load_problem("./scenarios/scenario1.txt")
    display_environment(problem, path=[problem.start1, problem.goal1])

