import math
import random
from collections import deque

import numpy as np

from src.grid.greedy_init import GreedyInitializer

EMPTY: int = 0
GRASS: int = 1
TENT: int = 2
TREE: int = 3
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]


class LocalSearch:
    def __init__(self, grid, row_limits, col_limits, max_iters=1000):
        self.grid = np.array(grid, dtype=np.int8)
        self.n = len(grid)
        self.m = len(grid[0])
        self.trees = [(r, c) for r in range(self.n) for c in range(self.m) if grid[r][c] == TREE]
        self.row_limits = row_limits
        self.col_limits = col_limits
        self.row_counts = [0] * self.n
        self.col_counts = [0] * self.m
        self.max_iters = max_iters
        self.last_tents = []
        self.max_score = 0
        self.tree_neighbors = {tree: self.get_neighbors(tree, set()) for tree in self.trees}
        self.eva = 0
        self.best_score = 0

    def norm_pair(self, a, b):
        return (a, b) if a <= b else (b, a)

    def evaluate(self, tents):
        score = 0
        self.row_counts = [0] * self.n
        self.col_counts = [0] * self.m
        tent_set = set(tents)
        visited_pairs = set()

        for x, y in tents:
            self.row_counts[x] += 1
            self.col_counts[y] += 1

            for dx, dy in NEIGHBORS:
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if (nx, ny) in tent_set:
                    pair = self.norm_pair((x, y), (nx, ny))
                    if pair not in visited_pairs:
                        score += 1
                        visited_pairs.add(pair)

        for i in range(self.n):
            score += abs(self.row_counts[i] - self.row_limits[i])
        for j in range(self.m):
            score += abs(self.col_counts[j] - self.col_limits[j])

        return score

    def delta_evaluate(self, tents, old_pos, new_pos, old_score, row_counts_old, col_counts_old):
        if old_pos == new_pos:
            return old_score
        row_counts = row_counts_old[:]
        col_counts = col_counts_old[:]

        x1, y1 = old_pos
        x2, y2 = new_pos

        delta = 0
        old_visited_pairs = set()

        tent_set = set(tents)

        # Удаляем штрафы соседства у старой палатки
        for dx, dy in NEIGHBORS:
            if dx == dy == 0:
                continue
            nx, ny = x1 + dx, y1 + dy
            if (nx, ny) in tent_set:
                pair = self.norm_pair((x1, y1), (nx, ny))
                if pair not in old_visited_pairs:
                    delta -= 1
                    old_visited_pairs.add(pair)

        tent_set.remove(old_pos)
        tent_set.add(new_pos)

        new_visited_pairs = set()
        # Добавляем штрафы соседства у новой палатки
        for dx, dy in NEIGHBORS:
            if dx == dy == 0:
                continue
            nx, ny = x2 + dx, y2 + dy
            if (nx, ny) in tent_set:
                pair = self.norm_pair((x2, y2), (nx, ny))
                if pair not in new_visited_pairs:
                    delta += 1
                    new_visited_pairs.add(pair)

        # Штраф за строки
        def row_penalty(i, delta_count):
            before = row_counts[i]
            return abs(before + delta_count - self.row_limits[i]) - abs(before - self.row_limits[i])

        def col_penalty(j, delta_count):
            before = col_counts[j]
            return abs(before + delta_count - self.col_limits[j]) - abs(before - self.col_limits[j])

        delta += row_penalty(x1, -1)
        delta += col_penalty(y1, -1)
        row_counts[x1] -= 1
        col_counts[y1] -= 1
        delta += row_penalty(x2, 1)
        delta += col_penalty(y2, 1)

        return old_score + delta

    def get_neighbors(self, tree_pos, occupied):
        neighbors = []
        for dx, dy in DIRS:
            nx, ny = tree_pos[0] + dx, tree_pos[1] + dy
            if (
                    0 <= nx < self.n and 0 <= ny < self.m
                    and self.grid[nx][ny] != TREE
                    and (nx, ny) not in occupied
            ):
                neighbors.append((nx, ny))
        return neighbors

    def smart_initialization(self):
        tents = [None] * len(self.trees)
        occupied = set()
        sorted_trees = sorted(self.trees, key=lambda t: len(self.tree_neighbors[t]))

        for tree in sorted_trees:
            idx = self.trees.index(tree)
            options = sorted(
                [pos for pos in self.tree_neighbors[tree] if pos not in occupied],
                key=lambda pos: sum(1 for dx, dy in NEIGHBORS if (pos[0] + dx, pos[1] + dy) in occupied)
            )
            if not options:
                return None
            chosen = options[0]
            tents[idx] = chosen
            occupied.add(chosen)

        return tents

    def random_kick(self, tents, strength=3):
        indices = random.sample(range(len(tents)), k=min(strength, len(tents)))
        for i in indices:
            neighbors = self.get_neighbors(self.trees[i], tents)
            if neighbors:
                tents[i] = random.choice(neighbors)

    def tabu_search0(self, row_limits, col_limits, tabu_size=100, max_stagnation=200):
        for _ in range(5):
            initializer = GreedyInitializer(self.grid.copy(), row_limits[:], col_limits[:])
            trees, current_tents = initializer.initialize()
            if current_tents and trees:
                break
        else:
            return None

        best_tents = current_tents[:]
        best_score = self.evaluate(best_tents)
        max_score = best_score
        current_score = best_score

        tabu_list = deque(maxlen=tabu_size)
        stagnation = 0

        # Используем numpy массивы для ускорения
        row_counts = np.array(self.row_counts, dtype=int)
        col_counts = np.array(self.col_counts, dtype=int)

        trees = list(enumerate(trees))  # [(i, tree), ...]

        for _ in range(self.max_iters * 3):
            if stagnation >= max_stagnation:
                break

            best_candidate = None
            best_candidate_score = float('inf')

            random.shuffle(trees)  # чтобы не застревать в паттернах

            for i, tree in trees:
                current_tent = current_tents[i]

                neighbors = self.get_neighbors(tree, current_tents)
                if not neighbors:
                    continue

                for candidate in neighbors:
                    if (i, candidate) in tabu_list and current_score <= best_score:
                        continue

                    # Инкрементальный пересчёт счётчиков
                    row_counts[current_tent[0]] -= 1
                    col_counts[current_tent[1]] -= 1
                    row_counts[candidate[0]] += 1
                    col_counts[candidate[1]] += 1

                    new_score = self.delta_evaluate(
                        current_tents,
                        current_tent,
                        candidate,
                        current_score,
                        row_counts,
                        col_counts
                    )

                    self.eva += 1

                    # Откат изменений
                    row_counts[current_tent[0]] += 1
                    col_counts[current_tent[1]] += 1
                    row_counts[candidate[0]] -= 1
                    col_counts[candidate[1]] -= 1

                    if not new_score:
                        continue

                    if new_score < best_candidate_score:
                        best_candidate_score = new_score
                        best_candidate = (i, candidate)

            if best_candidate is None:
                stagnation += 1
                continue

            i, candidate = best_candidate
            current_tent = current_tents[i]

            # Обновляем counts
            row_counts[current_tent[0]] -= 1
            col_counts[current_tent[1]] -= 1
            row_counts[candidate[0]] += 1
            col_counts[candidate[1]] += 1

            current_tents[i] = candidate
            current_score = best_candidate_score

            tabu_list.append((i, current_tent))

            if current_score < best_score:
                best_score = current_score
                best_tents = current_tents[:]
                stagnation = 0
            else:
                stagnation += 1

            if best_score == 0:
                print("Optimal solution found!")
                break

        self.last_tents = best_tents[:]
        self.max_score = max_score
        self.best_score = best_score
        return best_tents

    def tabu_search(self, row_limits, col_limits, tabu_size=100, max_stagnation=200):
        for _ in range(5):
            initializer = GreedyInitializer(self.grid.copy(), row_limits[:], col_limits[:])
            trees, current_tents = initializer.initialize()
            if current_tents and trees:
                break
        else:
            return None

        trees = [(i, tree) for i, tree in enumerate(trees)]

        best_tents = current_tents[:]
        best_score = self.evaluate(best_tents)
        max_score = best_score
        current_score = best_score

        tabu_list = deque(maxlen=tabu_size)
        stagnation = 0

        row_counts = self.row_counts[:]
        col_counts = self.col_counts[:]

        for _ in range(self.max_iters*10):
            if stagnation >= max_stagnation:
                break

            best_candidate = None
            best_candidate_score = float('inf')
            random.shuffle(trees)

            for i, tree in trees:
                current_tent = current_tents[i]
                neighbors = self.get_neighbors(tree, current_tents)

                for candidate in neighbors:
                    if (i, candidate) in tabu_list and current_score <= best_score:
                        continue

                    row_counts_tmp = row_counts[:]
                    col_counts_tmp = col_counts[:]

                    new_score = self.delta_evaluate(
                        current_tents,
                        current_tent,
                        candidate,
                        current_score,
                        row_counts_tmp,
                        col_counts_tmp
                    )

                    self.eva += 1
                    if not new_score:
                        continue

                    if new_score < best_candidate_score:
                        best_candidate_score = new_score
                        row_counts_tmp[current_tent[0]] -= 1
                        col_counts_tmp[current_tent[1]] -= 1
                        row_counts_tmp[candidate[0]] += 1
                        col_counts_tmp[candidate[1]] += 1
                        best_candidate = (i, candidate, row_counts_tmp, col_counts_tmp)

            if best_candidate is None:
                stagnation += 1
                continue

            i, candidate, row_counts, col_counts = best_candidate
            current_tent = current_tents[i]
            current_tents[i] = candidate
            current_score = best_candidate_score

            tabu_list.append((i, current_tent))

            if current_score < best_score:
                best_score = current_score
                best_tents = current_tents[:]
                stagnation = 0
            else:
                stagnation += 1

            if best_score == 0:
                print(f"Optimal solution found!")
                break

        self.last_tents = best_tents[:]
        self.max_score = max_score
        self.best_score = best_score
        return best_tents

    def tabu_search2(self, row_limits, col_limits, tabu_size=50):
        for _ in range(5):
            initializer = GreedyInitializer(self.grid.copy(), row_limits[:], col_limits[:])
            trees, current_tents = initializer.initialize()

            if current_tents and trees:
                break
        else:
            return None

        trees = [(i, tree) for i, tree in enumerate(trees)]

        best_tents = current_tents[:]
        best_score = self.evaluate(best_tents)
        max_score = best_score
        tabu_list = deque(maxlen=tabu_size)

        row_counts = self.row_counts[:]
        col_counts = self.col_counts[:]

        for _ in range(self.max_iters):
            best_candidate = None
            best_candidate_score = float('inf')
            random.shuffle(trees)

            for i, tree in trees:
                current_tent = current_tents[i]
                neighbors = self.get_neighbors(tree, current_tents)

                for candidate in neighbors:
                    if (i, candidate) in tabu_list:
                        continue

                    test_tents = current_tents[:]
                    test_tents[i] = candidate
                    score = self.delta_evaluate(current_tents, current_tent, candidate, best_score,
                                                row_counts[:], col_counts[:])
                    new_tents = current_tents[:]
                    new_tents[i] = candidate
                    if score != self.evaluate(new_tents):
                        print(f'new_score = {score} eva = {self.evaluate(new_tents)}')

                    self.eva += 1
                    if score < best_candidate_score:
                        row_counts[current_tent[0]] -= 1
                        col_counts[current_tent[1]] -= 1
                        row_counts[candidate[0]] += 1
                        col_counts[candidate[1]] += 1
                        best_candidate_score = score
                        self.best_score = score
                        best_candidate = (i, candidate, test_tents)

            if best_candidate is None:
                break

            i, candidate, new_tents = best_candidate
            tabu_list.append((i, candidate))
            current_tents = new_tents
            if best_candidate_score < best_score:
                best_score = best_candidate_score
                best_tents = current_tents[:]

            if best_score == 0:
                print(f'tents = {len(best_tents)}')
                print(f'trees = {len(self.trees)}')
                break

        self.last_tents = best_tents[:]
        self.max_score = max_score

        return best_tents

    def single_local_search(self, row_limits, col_limits):
        for _ in range(5):
            initializer = GreedyInitializer(self.grid.copy(), row_limits[:], col_limits[:])
            trees, tents = initializer.initialize()

            if tents and trees:
                break
        else:
            return None

        trees = [(i, tree) for i, tree in enumerate(trees)]

        best_score = self.evaluate(tents)
        self.eva += 1
        max_score = int(best_score)
        row_counts = self.row_counts[:]
        col_counts = self.col_counts[:]

        stagnation = 0
        for _ in range(self.max_iters*len(self.trees)):
            improved = False
            random.shuffle(trees)

            for i, tree in trees:
                current_tent = tents[i]
                neighbors = self.get_neighbors(tree, tents)

                for candidate in neighbors:
                    score = self.delta_evaluate(tents, current_tent, candidate, best_score,
                                                row_counts[:], col_counts[:])
                    self.eva += 1
                    if score < best_score:
                        stagnation = 0
                        row_counts[current_tent[0]] -= 1
                        col_counts[current_tent[1]] -= 1
                        row_counts[candidate[0]] += 1
                        col_counts[candidate[1]] += 1
                        tents[i] = candidate
                        best_score = score
                        improved = True
                        current_tent = candidate
                    else:
                        stagnation += 1

            if best_score == 0:
                self.last_tents = tents[:]
                self.max_score = max_score
                self.best_score = best_score
                return tents
            if not improved:
                break
            if stagnation > 100:  # не улучшается 100 шагов подряд
                break

        self.last_tents = tents[:]
        self.max_score = max_score
        self.best_score = best_score
        return tents

    def simulated_annealing(self, row_limits, col_limits, initial_temperature=10.0, cooling_rate=0.95,
                             min_temperature=0.01, max_stagnation=100):
        for _ in range(5):
            initializer = GreedyInitializer(self.grid.copy(), row_limits[:], col_limits[:])
            trees, current_tents = initializer.initialize()

            if current_tents and trees:
                break
        else:
            return None

        best_tents = current_tents[:]
        best_score = self.evaluate(best_tents)
        max_score = best_score
        current_score = best_score
        temperature = initial_temperature

        row_counts = self.row_counts[:]
        col_counts = self.col_counts[:]

        stagnation = 0

        for _ in range(self.max_iters*10):
            if temperature < min_temperature or stagnation > max_stagnation:
                break

            i = random.randint(0, len(self.trees) - 1)
            tree = self.trees[i]
            current_tent = current_tents[i]

            neighbors = self.get_neighbors(tree, current_tents)
            if not neighbors:
                continue

            candidate = random.choice(neighbors)

            new_score = self.delta_evaluate(
                current_tents,
                current_tent,
                candidate,
                current_score,
                row_counts,
                col_counts
            )
            new_tents = current_tents[:]
            new_tents[i] = candidate
            if new_score != self.evaluate(new_tents):
                print(f'new_score = {new_score} eva = {self.evaluate(new_tents)}')

            delta = new_score - current_score

            accept = False
            if delta < 0:
                accept = True
            else:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    accept = True

            if accept:
                current_tents[i] = candidate
                x1, y1 = current_tent
                x2, y2 = candidate
                row_counts[x1] -= 1
                col_counts[y1] -= 1
                row_counts[x2] += 1
                col_counts[y2] += 1
                current_score = new_score

                if new_score < best_score:
                    best_score = new_score
                    best_tents = current_tents[:]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            temperature *= cooling_rate

        self.last_tents = best_tents[:]
        self.max_score = max_score
        self.best_score = best_score
        return best_tents

    def variable_neighborhood_descent(self, row_limits, col_limits):
        def is_valid_move(pos, occupied, row_counts, col_counts):
            x, y = pos
            if row_counts[x] >= self.row_limits[x]:
                return False
            if col_counts[y] >= self.col_limits[y]:
                return False
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if (nx, ny) in occupied and (nx, ny) != pos:
                    return False
            return True

        best_overall = None
        best_score = float('inf')

        for _ in range(3):  # до 3 перезапусков инициализации
            initializer = GreedyInitializer(self.grid.copy(), row_limits[:], col_limits[:])
            trees, current_tents = initializer.initialize()

            if not trees or not current_tents:
                continue

            current_score = self.evaluate(current_tents)
            row_counts = self.row_counts[:]
            col_counts = self.col_counts

            neighborhood_size = 1
            max_neighborhood = 3
            indices = list(range(len(trees)))

            while neighborhood_size <= max_neighborhood:
                improved = False
                random.shuffle(indices)

                if neighborhood_size == 1:
                    for i in indices:
                        current_tent = current_tents[i]
                        temp_occupied = set(current_tents)
                        temp_occupied.remove(current_tent)

                        neighbors = self.get_neighbors(trees[i], temp_occupied)
                        for candidate in neighbors:
                            if candidate == current_tent:
                                continue
                            if not is_valid_move(candidate, temp_occupied, row_counts, col_counts):
                                continue
                            # new_score = self.evaluate(current_tents)
                            new_score = self.delta_evaluate(
                                current_tents, current_tent, candidate, current_score,
                                row_counts[:], col_counts[:]
                            )
                            self.eva += 1
                            if new_score is None:
                                continue
                            if new_score < current_score:
                                x1, y1 = current_tent
                                x2, y2 = candidate
                                row_counts[x1] -= 1
                                col_counts[y1] -= 1
                                row_counts[x2] += 1
                                col_counts[y2] += 1
                                current_tents[i] = candidate
                                current_score = new_score
                                improved = True
                                current_tent = candidate
                        if improved:
                            break

                elif neighborhood_size in (2, 3):
                    from itertools import combinations, product
                    MAX_GROUPS = 100
                    MAX_PRODUCTS = 50

                    all_groups = list(combinations(indices, neighborhood_size))
                    random.shuffle(all_groups)
                    limited_groups = all_groups[:MAX_GROUPS]

                    for group in limited_groups:
                        current_positions = [current_tents[i] for i in group]
                        temp_occupied = set(current_tents) - set(current_positions)

                        neighbor_lists = [
                            [p for p in self.get_neighbors(trees[i], temp_occupied)
                             if is_valid_move(p, temp_occupied, self.row_counts, self.col_counts)]
                            for i in group
                        ]
                        if any(len(lst) == 0 for lst in neighbor_lists):
                            continue

                        for j, candidate_group in enumerate(product(*neighbor_lists)):
                            if j >= MAX_PRODUCTS:
                                break
                            if len(set(candidate_group)) < len(candidate_group):
                                continue

                            delta_score = current_score
                            temp_row_counts = row_counts[:]
                            temp_col_counts = col_counts[:]
                            valid = True

                            for idx, new_pos in zip(group, candidate_group):
                                old_pos = current_tents[idx]
                                # delta_score = self.evaluate(current_tents)
                                delta_score = self.delta_evaluate(
                                    current_tents,
                                    old_pos,
                                    new_pos,
                                    delta_score,
                                    temp_row_counts,
                                    temp_col_counts
                                )
                                if delta_score is None:
                                    valid = False
                                    break

                            if valid and delta_score < current_score:
                                for idx, new_pos in zip(group, candidate_group):
                                    old_pos = current_tents[idx]
                                    x1, y1 = old_pos
                                    x2, y2 = new_pos
                                    self.row_counts[x1] -= 1
                                    self.col_counts[y1] -= 1
                                    self.row_counts[x2] += 1
                                    self.col_counts[y2] += 1
                                    current_tents[idx] = new_pos

                                current_score = delta_score
                                improved = True
                                break
                        if improved:
                            break

                if improved:
                    neighborhood_size = 1
                else:
                    neighborhood_size += 1

                if current_score == 0:
                    self.best_score = 0
                    return current_tents

            if current_score < best_score:
                best_score = current_score
                best_overall = current_tents[:]

        self.best_score = best_score
        return best_overall

    def solve(self, row_limits, col_limits, restarts=10, method="local", random_init=True):
            best_global_score = float('inf')
            best_global_solution = None
            best_global_evaluation_count = 0
            best_global_max_score = 0

            for attempt in range(restarts):
                # print(f"\n🔁 Restart #{attempt + 1}, method={method}, best_score {best_global_score}")

                if method == "local":
                    solution = self.single_local_search(row_limits[:], col_limits[:])
                elif method == "annealing":
                    solution = self.simulated_annealing(row_limits[:], col_limits[:])
                elif method == "tabu":
                    solution = self.tabu_search(row_limits, col_limits)
                elif method == "vnd":
                    solution = self.variable_neighborhood_descent(row_limits[:], col_limits[:])
                else:
                    raise ValueError("Unknown method")

                # Если решение найдено
                if solution is not None:
                    if self.best_score == 0:
                        return solution, self.best_score, self.max_score, self.eva, attempt
                    elif self.best_score < best_global_score:
                        best_global_score = self.best_score
                        best_global_solution = solution[:]
                        best_global_evaluation_count = self.eva
                        best_global_max_score = self.max_score

            return best_global_solution, best_global_score, best_global_max_score, best_global_evaluation_count, restarts

    def get_trees(self, step):
        trees = []
        for i, tree in enumerate(self.trees):
            trees.append((i, tree))
        return trees[step:] + trees[:step]
