import time

import statistics

from src.algorithms.backtracking_solver import BacktrackSolver
from src.grid.grid_utils import print_grid
from src.grid.grid import GridOptim
from src.algorithms.metaheuristics import LocalSearch
from src.algorithms.ilp_solver import ilp_solver


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # время начала
        result = func(*args, **kwargs)
        end_time = time.perf_counter()    # время окончания
        elapsed_time = end_time - start_time
        print(f"The task {func.__name__} took {elapsed_time:.7f} seconds to complete.")
        return result, elapsed_time
        # return result
    return wrapper


def main():
    # grid, row_constraints, col_constraints = example_7x7()
    # print_grid(grid, row_constraints, col_constraints)
    # solver(grid, row_constraints, col_constraints)


    # test_solver(grid, row_constraints, col_constraints)
    # grid = Grid(9)
    #
    # print()
    # print_grid(grid.grid, grid.row_constraints, grid.col_constraints)
    # test_solver(grid.grid, grid.row_constraints, grid.col_constraints)

    # grid = [
    #     [0, 0, 0, 3],
    #     [0, 0, 0, 0],
    #     [3, 0, 0, 0],
    #     [0, 0, 3, 0]
    # ]
    # row_limits = [1, 1, 1, 0]
    # col_limits = [1, 0, 2, 0]
    # grid, row_constraints, col_constraints = example_5x5_test()
    # print_grid(grid, row_constraints, col_constraints)

    # grid = Grid(15)
    # grid, row_constraints, col_constraints = grid.grid, grid.row_constraints, grid.col_constraints


    # grid, row_constraints, col_constraints = example_8x8_hard()
    # print_grid(grid, row_constraints, col_constraints)
    # test_solver(grid, row_constraints, col_constraints)
    # print('-' * 20)
    # print('-' * 20)
    # print('-' * 20)
    # print('-' * 20)
    # grid_copy = list(grid)
    # solver_backtracking = BacktrackHardSolver(grid_copy, row_constraints, col_constraints)
    # solver_backtracking.solve()
    # print_grid(grid_copy, row_constraints, col_constraints)





    # grid = GridOptim(20)
    # grid_copy = grid.grid.copy()
    #
    # test_solver(grid.grid, grid.row_constraints, grid.col_constraints)
    # print('-' * 20)
    # print('-' * 20)
    # print('-' * 20)
    # print('-' * 20)
    #

    # back = BacktrackSolverOptim
    # grid = GridOptim(12)
    # grid2 = [[int(grid.grid[i, j]) for j in range(12)] for i in range(12)]
    #
    # grid_copy = grid.grid.copy()
    # grid, row_constraints, col_constraints = grid_copy, grid.row_constraints, grid.col_constraints

    # grid, row_constraints, col_constraints = example_8x8()
    # # solver_backtracking = TentsAndTreesSolver(grid, row_constraints, col_constraints)
    # grid, row_constraints, col_constraints = grid_copy, grid.row_constraints, grid.col_constraints
    # grid = np.array(grid, dtype=np.int8)


    # бэктрекинг

    # solver_backtracking = BacktrackSolver(grid, row_constraints, col_constraints)
    # start_time = time.perf_counter()
    # solver_backtracking.solve()
    # end_time = time.perf_counter()
    #
    # elapsed_time = end_time - start_time
    # print('-' * 20)
    # print(f"The task took {elapsed_time:.7f} seconds to complete.")
    # print('-' * 20)
    # print_grid(grid, row_constraints, col_constraints)
    #



    # grid = GridOptim(25)
    # grid_copy = grid.grid.copy()
    # grid, row_constraints, col_constraints = grid_copy, grid.row_constraints, grid.col_constraints
    # solver_backtracking = BacktrackSolverOptim(grid, row_constraints[:], col_constraints[:])
    #
    # start_time = time.perf_counter()
    # solver_backtracking.solve()
    # end_time = time.perf_counter()
    #
    # # Вывод результата
    # if solver_backtracking.solution_grid is not None:
    #     print("Итоговая сетка:")
    #     print_grid(solver_backtracking.solution_grid, row_constraints, col_constraints)
    # else:
    #     print("Решение не найдено")




    # elapsed_time = end_time - start_time
    # print('-' * 20)
    # print(f"The task took {elapsed_time:.7f} seconds to complete.")
    # print('-' * 20)

    # solver_back = BacktrackOptim(grid, row_constraints, col_constraints)


    # solver_back = BacktrackOptim(grid, row_constraints, col_constraints)
    # solver_back = BacktrackOptimThree(grid, row_constraints, col_constraints)
    # print(grid)
    # print(row_constraints)
    # print(col_constraints)
    # solver_back.solve()
    # # solver_back.solve_parallel()
    #
    #
    # # Вывод результата
    # if solver_back.solution_grid is not None:
    #     print("Итоговая сетка:")
    #     print_grid(solver_back.solution_grid, row_constraints, col_constraints)
    # else:
    #     print("Решение не найдено")

# #     # grid, row_constraints, col_constraints = example_12x12()
# # #     local search
#     local_search = LocalSearchSolver(grid[:], row_constraints, col_constraints)
#
#     # tents, best_score = local_search.solve_with_restarts(500, method="annealing")
# #     # tents, best_score = local_search.solve_with_annealing_and_tabu(restarts=500)
#     start_time = time.perf_counter()
#     # tents, best_score = local_search.solve_with_restarts(1000, method="local")
#     # tents, best_score = local_search.solve_with_restarts(10000, method="annealing")
#     # tents, best_score = local_search.solve_with_restarts(1000, method="tabu")
#     # tents, best_score = local_search.solve_with_restarts(10000, method="hybrid")
#     tents, best_score = local_search.solve_with_restarts(1000, method="vnd")
#     end_time = time.perf_counter()
# #     local_search = LocalSearchSolverOptim(grid[:], row_constraints, col_constraints)
# #     grid = GridOptim(16)
#     # grid_copy = grid.grid.copy()
#     # grid, row_constraints, col_constraints = grid.grid, grid.row_constraints, grid.col_constraints
#
#
#     # local_search = LocalOptimTwo(grid, row_constraints, col_constraints)
#     # tents, best_score = local_search.solve_with_restarts(500)
#
#
#
#     # local_search = LocalSearchSolverOptim(
#     #     grid, row_constraints, col_constraints,
#     #     max_iters=6000,
#     #     initial_temperature=100.0,
#     #     base_cooling_rate=0.995,
#     #     swap_k_range=(1, 3),
#     #     patience=120,
#     #     tabu_tenure=300  # размер Tabu List
#     # )
#     # tents, best_score = local_search.solve(500)
#     #
#     if tents:
#         print(f"Решение найдено, best_score {best_score}")
#         for tent in tents:
#             grid[tent[0]][tent[1]] = 2
#         print_grid(grid, row_constraints, col_constraints)
#     else:
#         print(f"Локальный минимум, best_score {best_score}")
#
#     elapsed_time = end_time - start_time
#     print('-' * 20)
#     print(f"The task took {elapsed_time:.7f} seconds to complete.")
#     print('-' * 20)
#     grid = GridOptim(10)
#     grid_init = grid.grid_init
    # print_init_grid(grid.grid, grid.row_constraints, grid.col_constraints)
    # grid, row_constraints, col_constraints = grid.grid, grid.row_constraints, grid.col_constraints

    # test_local_search_solver(grid, row_constraints, col_constraints)

    # ilp_time =  ilp(grid.copy(), row_constraints[:], col_constraints[:])
    # boo, back_time = backtrack(grid.copy(), row_constraints[:], col_constraints[:])
    # ls_eva,ls_time = local_search(grid.copy(), row_constraints[:], col_constraints[:])
    # vnd_eva, vnd_time =  vnd(grid.copy(), row_constraints[:], col_constraints[:])
    # sa_eva, sa_time = annealing(grid.copy(), row_constraints[:], col_constraints[:])
    # hybrid(grid, row_constraints, col_constraints)
    # tabu_eva, tabu_time = tabu(grid.copy(), row_constraints[:], col_constraints[:])
    # print_grid(grid_init, row_constraints, col_constraints)

    # print(f'ilp: {ilp_time[1]:.4f}')
    # print(f'backtracking: {back_time:.4f}, {boo}')
    # print(f'local search: {ls_time:.4f}, {ls_eva:.4f}')
    # print(f'vnd: {vnd_time:.4f}, {vnd_eva:.4f}')
    # print(f'sa: {sa_time:.4f}, {sa_eva:.4f}')
    # print(f'tabu: {tabu_time:.4f}, {tabu_eva:.4f}')

    # test_10x10(100)
    # initializer = GreedyInitializer(grid, row_constraints, col_constraints)
    # links = initializer.initialize()
    # print_grid_with_tents(links.values(), grid, row_constraints, col_constraints)
    # print_grid(new_grid, row_constraints, col_constraints)

    test_algorithms(10, 25, 25 )


def test_algorithms(n, size_n, size_m):
    # ilp_times = [779.7346605, 681.1402931, 1069.9386659, 395.7273931, 1394.8691903, 2248.45]
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
        # grid = [
        #     [0, 3, 0, 0],
        #     [0, 0, 0, 0],
        #     [3, 0, 0, 3],
        #     [0, 0, 0, 0]
        # ]
        # row_constraints = [1, 1, 1, 0]
        # col_constraints = [1, 1, 0, 1]
        #
        # ilp_time = ilp(grid.copy(), row_constraints[:], col_constraints[:])
        # ilp_times.append(ilp_time[1])
        #
        boo, back_time = backtrack(grid.copy(), row_constraints[:], col_constraints[:])
        back_times.append(back_time)
        if boo:
            back_cnt_solve += 1
        #
        # ls_eva, ls_time = local_search(grid.copy(), row_constraints[:], col_constraints[:])
        # ls_times.append(ls_time)
        # ls_evaluate.append(ls_eva)
        # if ls_eva == 0:
        #     ls_cnt_solve += 1

        # vnd_eva, vnd_time = vnd(grid.copy(), row_constraints[:], col_constraints[:])
        # vnd_times.append(vnd_time)
        # vnd_evaluate.append(vnd_eva)
        # if vnd_eva == 0:
        #     vnd_cnt_solve += 1
        # #
        # sa_eva, sa_time = annealing(grid.copy(), row_constraints[:], col_constraints[:])
        # sa_times.append(sa_time)
        # sa_evaluate.append(sa_eva)
        # if sa_eva == 0:
        #     sa_cnt_solve += 1

        # tabu_eva, tabu_time = tabu(grid.copy(), row_constraints[:], col_constraints[:])
        # tabu_eva, tabu_time = tabu2(grid.copy(), row_constraints[:], col_constraints[:])
        # tabu_times.append(tabu_time)
        # tabu_evaluate.append(tabu_eva)
        # if tabu_eva in [0, 1, 2]:
        #     tabu_cnt_solve += 1

    # print_algorithm_times('ilp', ilp_times, None, n, [0] * n)
    print_algorithm_times('backtracking', back_times, back_cnt_solve, n, [0] * n)
    # print_algorithm_times('local', ls_times, ls_cnt_solve, n, ls_evaluate)
    # print_algorithm_times('vnd', vnd_times, vnd_cnt_solve, n, vnd_evaluate)
    # print_algorithm_times('annealing', sa_times, sa_cnt_solve, n, sa_evaluate)
    # print_algorithm_times('tabu', tabu_times, tabu_cnt_solve, n, tabu_evaluate)


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


def print_grid_with_tents(tents, grid, row_limits, col_limits):
    print('-' * 50)
    grid = grid.copy()
    for tent in tents:
        grid[tent[0]][tent[1]] = 2
    print_grid(grid, row_limits, col_limits)


def test_local_search_solver(grid, row_constraints, col_constraints):
    local_search_solver = LocalSearch(grid, row_constraints, col_constraints)
    local_search_solver.solve()


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
    # grid = GridOptim(15)
    # grid, row_constraints, col_constraints = grid.grid, grid.row_constraints, grid.col_constraints
    local_search_solver = LocalSearch(grid[:], row_constraints, col_constraints)
    res = (
        local_search_solver.solve(row_constraints, col_constraints, 100, method="local"))
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]
    # local_search_solver = LocalSearchSolver(grid[:], row_constraints, col_constraints)
    # res = (
    #     local_search_solver.solve_with_restarts(row_constraints, col_constraints, 100, method="local"))
    # tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]
    # if tents:
    #     print(f"Решение найдено, best_score {best_score}")
    #     print(f'max_score = {max_score}')
    #     print(f'evaluate_cnt = {eva}')
    #     print(f'restars = {cnt_restarts}')
    #     for tent in tents:
    #         grid[tent[0]][tent[1]] = 2
    #     print_grid(grid, row_constraints, col_constraints)
    # else:
    #     print(f"Локальный минимум, best_score {best_score}")

    return best_score


@timer
def annealing(grid, row_constraints, col_constraints):
    # local_search_solver = LocalSearchSolver(grid[:], row_constraints, col_constraints)
    # res =\
    #     local_search_solver.solve_with_restarts(row_constraints, col_constraints,100, method="annealing")
    # tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]
    local_search_solver = LocalSearch(grid[:], row_constraints, col_constraints)
    res = \
        local_search_solver.solve(row_constraints, col_constraints, 150, method="annealing")
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]
    # if tents:
    #     print(f"Решение найдено, best_score {best_score}")
    #     print(f'max_score = {max_score}')
    #     print(f'evaluate_cnt = {eva}')
    #     print(f'restars = {cnt_restarts}')
    #     for tent in tents:
    #         grid[tent[0]][tent[1]] = 2
    #     print_grid(grid, row_constraints, col_constraints)
    # else:
    #     print(f"Локальный минимум, best_score {best_score}")

    return  best_score

# @timer
# def hybrid(grid, row_constraints, col_constraints):
#     local_search_solver = LocalSearchSolver(grid[:], row_constraints, col_constraints)
#     tents, best_score, max_score, eva, cnt_restarts = \
#         local_search_solver.solve_with_restarts(row_constraints, col_constraints, 100, method="hybrid")
#
#     print(f"Решение найдено, best_score {best_score}")
#     print(f'max_score = {max_score}')
#     for tent in tents:
#         grid[tent[0]][tent[1]] = 2
#     print_grid(grid, row_constraints, col_constraints)


@timer
def tabu2(grid, row_constraints, col_constraints):
    local_search_solver = LocalSearch(grid[:], row_constraints, col_constraints)
    res = (
        local_search_solver.solve(row_constraints, col_constraints,20, method="tabu"))
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]
    # if tents:
    #     print(f"Решение найдено, best_score {best_score}")
    #     print(f'max_score = {max_score}')
    #     print(f'score = {local_search_solver.evaluate(tents)}')
    #     print(f'evaluate_cnt = {eva}')
    #     print(f'restars = {cnt_restarts}')
    #     for tent in tents:
    #         grid[tent[0]][tent[1]] = 2
    #     print_grid(grid, row_constraints, col_constraints)
    # else:
    #     print(f"Локальный минимум, best_score {best_score}")

    return  best_score


@timer
def vnd(grid, row_constraints, col_constraints):
    # local_search_solver = LocalSearchSolver(grid[:], row_constraints, col_constraints)
    # res =\
    #     local_search_solver.solve_with_restarts(row_constraints, col_constraints,100, method="vnd")
    # tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]
    local_search_solver = LocalSearch(grid[:], row_constraints, col_constraints)
    res = \
        local_search_solver.solve(row_constraints, col_constraints, 50, method="vnd")
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]
    # if tents:
    #     print(f"Решение найдено, best_score {best_score}")
    #     print(f'max_score = {max_score}')
    #     print(f'evaluate_cnt = {eva}')
    #     print(f'restars = {cnt_restarts}')
    #     for tent in tents:
    #         grid[tent[0]][tent[1]] = 2
    #     print_grid(grid, row_constraints, col_constraints)
    # else:
    #     print(f"Локальный минимум, best_score {best_score}")

    return  best_score


def print_init_grid(grid, row_constraints, col_constraints):
    grid2 = []
    for i in range(len(grid)):
        grid2.append([])
        for j in range(len(grid)):
            grid2[i].append(int(grid[i, j]))

    # grid2 = [[int(grid[i, j]) for j in range(len(grid))] for i in range(len(grid))]
    print('[')
    print(*grid2, sep=',\n')
    print(']')
    print(row_constraints)
    print(col_constraints)


def test_print(grid, row_constraints, col_constraints):
    print(f'grid = ', end='')
    print_grid_array(grid)
    print(f'row = {row_constraints}')
    print(f'col = {col_constraints}')


def test_10x10(n):
    random_best_score = 0
    random_max_score = 0
    random_eva = 0
    random_cnt_restarts = 0
    random_norm = 0.0
    random_time = 0.0
    random_cnt_optim = 0

    zhadn_best_score = 0
    zhadn_max_score = 0
    zhadn_eva = 0
    zhadn_cnt_restarts = 0
    zhadn_norm = 0.0
    zhadn_time = 0.0
    zhadn_cnt_optim = 0

    for i in range(n):
        grid = GridOptim(7)
        grid, row_constraints, col_constraints = grid.grid, grid.row_constraints, grid.col_constraints
        # test_print(grid, row_constraints, col_constraints)
        print('случайная генерация')

        random_result, rand_time = local_search(grid, row_constraints, col_constraints)
        best_score, max_score, eva, cnt_restarts = random_result[0], random_result[1],random_result[2], random_result[3]
        random_best_score += best_score
        random_max_score += max_score
        random_eva +=  eva
        random_cnt_restarts += cnt_restarts
        random_norm += (1-best_score/max_score)
        random_time += rand_time
        if best_score == 0:
            random_cnt_optim += 1

        print('жадная генерация')
        zhadn_result, zh_time = local_search(grid, row_constraints, col_constraints)
        best_score, max_score, eva, cnt_restarts = zhadn_result[0], zhadn_result[1], zhadn_result[2], zhadn_result[
            3]
        zhadn_best_score += best_score
        zhadn_max_score += max_score
        zhadn_eva += eva
        zhadn_cnt_restarts += cnt_restarts
        zhadn_norm += (1-best_score/max_score)
        zhadn_time += zh_time
        if best_score == 0:
            zhadn_cnt_optim += 1

    print('Среднее для рандома: ')
    print(f'best_score = {random_best_score/n:.7f}')
    print(f'max_score = {random_max_score/n:.7f}')
    print(f'random_eva = {random_eva/n:.7f}')
    print(f'restars = {random_cnt_restarts/n:.7f}')
    print(f'norm = {random_norm/n:.7f}')
    print(f'time = {random_time/n:.7f}')
    print(f'cnt_optim = {random_cnt_optim}')

    print('\nСреднее для жадного: ')
    print(f'best_score = {zhadn_best_score / n:.7f}')
    print(f'max_score = {zhadn_max_score / n:.7f}')
    print(f'random_eva = {zhadn_eva / n:.7f}')
    print(f'restars = {zhadn_cnt_restarts / n:.7f}')
    print(f'norm = {zhadn_norm / n:.7f}')
    print(f'time = {zhadn_time/n:.7f}')
    print(f'cnt_optim = {zhadn_cnt_optim}')


def print_grid_array(grid):
    print("[")
    for i, row in enumerate(grid):
        line = "    [" + ", ".join(str(cell) for cell in row) + "]"
        if i < len(grid) - 1:
            line += ","
        print(line)
    print("]")


if __name__ == '__main__':
    main()
