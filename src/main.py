import time

import statistics
from TentsAndTrees.src.algorithms.backtracking_solver import BacktrackSolver
from TentsAndTrees.src.algorithms.ilp_solver import ilp_solver
from TentsAndTrees.src.algorithms.metaheuristics import LocalSearch
from TentsAndTrees.src.grid.grid import GridOptim


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # время начала
        result = func(*args, **kwargs)
        end_time = time.perf_counter()  # время окончания
        elapsed_time = end_time - start_time
        print(f"The task {func.__name__} took {elapsed_time:.7f} seconds to complete.")
        return result, elapsed_time
        # return result

    return wrapper


def main():
    test_algorithms(50, 5, 5)


def test_algorithms(n, size_n, size_m):
    ilp_times = []

    back_times = []
    back_cnt_solve = 0

    ls_times = []
    ls_cnt_solve = 0
    ls_evaluate = []

    vnd_times = []
    vnd_cnt_solve = 0
    vnd_evaluate = []

    sa_times = []
    sa_cnt_solve = 0
    sa_evaluate = []

    tabu_times = []
    tabu_cnt_solve = 0
    tabu_evaluate = []

    for _ in range(n):
        grid = GridOptim(size_n, size_m, "easy")
        grid, row_constraints, col_constraints = grid.grid, grid.row_constraints, grid.col_constraints

        boo, back_time = backtrack(grid.copy(), row_constraints[:], col_constraints[:])
        back_times.append(back_time)
        if boo:
            back_cnt_solve += 1

        ls_eva, ls_time = local_search(grid.copy(), row_constraints[:], col_constraints[:])
        ls_times.append(ls_time)
        ls_evaluate.append(ls_eva)
        if ls_eva == 0:
            ls_cnt_solve += 1

        vnd_eva, vnd_time = vnd(grid.copy(), row_constraints[:], col_constraints[:])
        vnd_times.append(vnd_time)
        vnd_evaluate.append(vnd_eva)
        if vnd_eva == 0:
            vnd_cnt_solve += 1
        #
        sa_eva, sa_time = annealing(grid.copy(), row_constraints[:], col_constraints[:])
        sa_times.append(sa_time)
        sa_evaluate.append(sa_eva)
        if sa_eva == 0:
            sa_cnt_solve += 1

        tabu_eva, tabu_time = tabu(grid.copy(), row_constraints[:], col_constraints[:])
        tabu_times.append(tabu_time)
        tabu_evaluate.append(tabu_eva)
        if tabu_eva in [0, 1, 2]:
            tabu_cnt_solve += 1

    print_algorithm_times('ilp', ilp_times, None, n, [0] * n)
    print_algorithm_times('backtracking', back_times, back_cnt_solve, n, [0] * n)
    print_algorithm_times('local', ls_times, ls_cnt_solve, n, ls_evaluate)
    print_algorithm_times('vnd', vnd_times, vnd_cnt_solve, n, vnd_evaluate)
    print_algorithm_times('annealing', sa_times, sa_cnt_solve, n, sa_evaluate)
    print_algorithm_times('tabu', tabu_times, tabu_cnt_solve, n, tabu_evaluate)


def print_algorithm_times(name, times, cnt_solve, n, evaluate):
    print('-' * 50)
    print(f'{name}')
    print(f'min_time = {min(times):.4f}')
    print(f'max_time = {max(times):.4f}')
    print(f'avg_time = {sum(times) / n:.4f}')

    if not name == 'ilp':
        print(f'percent_solve = {cnt_solve / n}')

    avg = statistics.mean(times)
    std_dev = statistics.stdev(times)  # стандартное отклонение

    print(f"time = {avg:.4f} ± {std_dev:.4f} секунд")

    print(f'times_{name} = {times}')
    print(f'scores_{name} = {evaluate}')


@timer
def ilp(grid, row_constraints, col_constraints):
    ilp_time = ilp_solver(grid, row_constraints, col_constraints)
    return ilp_time


@timer
def backtrack(grid, row_constraints, col_constraints):
    solver_backtracking = BacktrackSolver(grid, row_constraints, col_constraints)
    boo = solver_backtracking.solve()
    return boo


@timer
def local_search(grid, row_constraints, col_constraints, random_init=True):
    local_search_solver = LocalSearch(grid[:], row_constraints, col_constraints)
    res = (
        local_search_solver.solve(row_constraints, col_constraints, 100, method="local"))
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]

    return best_score


@timer
def annealing(grid, row_constraints, col_constraints):
    local_search_solver = LocalSearch(grid[:], row_constraints, col_constraints)
    res = \
        local_search_solver.solve(row_constraints, col_constraints, 150, method="annealing")
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]

    return best_score


@timer
def tabu(grid, row_constraints, col_constraints):
    local_search_solver = LocalSearch(grid[:], row_constraints, col_constraints)
    res = (
        local_search_solver.solve(row_constraints, col_constraints, 20, method="tabu"))
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]

    return best_score


@timer
def vnd(grid, row_constraints, col_constraints):
    local_search_solver = LocalSearch(grid[:], row_constraints, col_constraints)
    res = \
        local_search_solver.solve(row_constraints, col_constraints, 50, method="vnd")
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]

    return best_score


def print_init_grid(grid, row_constraints, col_constraints):
    grid2 = []
    print('{')
    print('"grid":', end='')
    for i in range(len(grid)):
        grid2.append([])
        for j in range(len(grid)):
            grid2[i].append(int(grid[i, j]))

    print('[')
    print(*grid2, sep=',\n')
    print('],')
    print(f'"row_constraints": {row_constraints},')
    print(f'"col_constraints": {col_constraints}')
    print('},')


if __name__ == '__main__':
    main()
