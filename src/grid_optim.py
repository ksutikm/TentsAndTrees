# import numpy as np
# import random
#
# _EMPTY = 0
# _TENT = 2
# _TREE = 3
# _MAX_TRIES = 100
#
# _FOUR_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# _EIGHT_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
#                      (0, 1), (1, 1), (1, 0),
#                      (1, -1), (0, -1)]
#
#
# class BadGenerationException(Exception):
#     pass
#
#
# class GridOptim:
#     def __init__(self, size: int):
#         self.size = size
#         self.num_trees = int(random.uniform(0.2, 0.25) * size**2)
#         generation = False
#
#         while not generation:
#             try:
#                 self.grid = np.zeros((size, size), dtype=int)
#                 self.links = {}  # дерево -> палатка
#                 self._place_tents_and_trees()
#                 self.row_constraints, self.col_constraints = self._get_row_col_constraints()
#                 print_grid(self.grid, self.row_constraints, self.col_constraints)
#                 self._remove_tents()
#                 generation = True
#             except BadGenerationException:
#                 pass
#
#     def _place_tents_and_trees(self):
#         all_cells = [(x, y) for x in range(self.size) for y in range(self.size)]
#         random.shuffle(all_cells)
#
#         trees_placed = 0
#
#         for x, y in all_cells:
#             if trees_placed >= self.num_trees:
#                 break
#             if self.grid[x][y] != _EMPTY:
#                 continue
#
#             neighbours = self._get_neighbours(x, y, k=4)
#             random.shuffle(neighbours)
#
#             for nx, ny in neighbours:
#                 if self.grid[nx][ny] != _EMPTY:
#                     continue
#                 if self._has_adjacent_tent(nx, ny):
#                     continue
#
#                 self.grid[x][y] = _TREE
#                 self.grid[nx][ny] = _TENT
#                 self.links[(x, y)] = (nx, ny)
#                 trees_placed += 1
#                 break
#
#         if trees_placed < self.num_trees:
#             raise BadGenerationException("Не удалось разместить все деревья.")
#
#     def _has_adjacent_tent(self, x, y):
#         for nx, ny in self._get_neighbours(x, y, k=8):
#             if self.grid[nx][ny] == _TENT:
#                 return True
#         return False
#
#     def _get_neighbours(self, x: int, y: int, k=4):
#         lst_vars = []
#         neighbours = _FOUR_NEIGHBOURS if k == 4 else _EIGHT_NEIGHBOURS
#
#         for dx, dy in neighbours:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < self.size and 0 <= ny < self.size:
#                 if k == 4 and self.grid[nx][ny] == _EMPTY:
#                     lst_vars.append((nx, ny))
#                 if k == 8 and self.grid[nx][ny] == _TENT:
#                     lst_vars.append((nx, ny))
#
#         return lst_vars
#
#     def _get_row_col_constraints(self):
#         row_constraints = [int(np.sum(row == _TENT)) for row in self.grid]
#         col_constraints = [int(np.sum(col == _TENT)) for col in self.grid.T]
#         return row_constraints, col_constraints
#
#     def _remove_tents(self):
#         for x in range(self.size):
#             for y in range(self.size):
#                 if self.grid[x][y] == _TENT:
#                     self.grid[x][y] = _EMPTY
#
#     def print_grid(self):
#         for row in self.grid:
#             print(" ".join(str(cell) for cell in row))
#         print("Row constraints:", self.row_constraints)
#         print("Col constraints:", self.col_constraints)
#
#
# def print_grid(grid, row_constraints, col_constraints, tent=_TENT):
#     size = len(grid)
#     print(' ' * 6, end='')
#     print(*col_constraints, sep=' ' * 2)
#     print('_' * (size * 4))
#
#     for row in range(size):
#         print(f'{row_constraints[row]} |', sep='', end=' ')
#         for col in range(size):
#             if grid[row][col] == tent:
#                 print("⛺", sep='', end=' ')
#             elif grid[row][col] == _TREE:
#                 print("🌲", sep='', end=' ')
#             else:
#                 print("◻️", sep='', end=' ')
#         print()
#

import numpy as np
import random

_EMPTY = 0
_TENT = 2
_TREE = 3

_FOUR_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_EIGHT_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
                     (0, 1), (1, 1), (1, 0),
                     (1, -1), (0, -1)]


class BadGenerationException(Exception):
    pass


class GridOptim:
    def __init__(self, size: int):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.links = {}  # дерево -> палатка
        self.empty_cells = {(x, y) for x in range(size) for y in range(size)}
        self._cache_all_neighbours()

        # for ratio in np.arange(0.23, 0.22, -0.01):
        for ratio in np.arange(0.25, 0.17, -0.01):
        # for ratio in np.arange(0.22, 0.19, -0.01):
            self.num_trees = int(ratio * size ** 2)
            self._reset_grid()
            success = self._place_tents_and_trees()
            if success and self._validate_grid():
                break
        else:
            raise BadGenerationException("Невозможно сгенерировать даже при низкой плотности.")

        # while True:
        #     self.num_trees = int(0.23 * size ** 2)
        #     self._reset_grid()
        #     success = self._place_tents_and_trees()
        #     if success and self._validate_grid():
        #         break

        self.row_constraints, self.col_constraints = self._get_row_col_constraints()
        print_grid(self.grid, self.row_constraints, self.col_constraints)
        self.grid_init = self.grid.copy()
        self._remove_tents()

    def _reset_grid(self):
        self.grid.fill(_EMPTY)
        self.links.clear()
        self.empty_cells = {(x, y) for x in range(self.size) for y in range(self.size)}

    def _cache_all_neighbours(self):
        self.cached_4 = {}
        self.cached_8 = {}
        for x in range(self.size):
            for y in range(self.size):
                self.cached_4[(x, y)] = self._compute_neighbours(x, y, _FOUR_NEIGHBOURS)
                self.cached_8[(x, y)] = self._compute_neighbours(x, y, _EIGHT_NEIGHBOURS)

    def _compute_neighbours(self, x, y, directions):
        neighbours = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbours.append((nx, ny))
        return neighbours

    def _place_tents_and_trees(self):
        trees_placed = 0
        pairs = self._generate_valid_pairs()
        random.shuffle(pairs)

        used_tents = set()

        for (x, y), (nx, ny) in pairs:
            if trees_placed >= self.num_trees:
                return True
            if (x, y) not in self.empty_cells or (nx, ny) not in self.empty_cells:
                continue
            if (nx, ny) in used_tents:
                continue
            if self._has_adjacent_tent(nx, ny):
                continue

            self.grid[x][y] = _TREE
            self.grid[nx][ny] = _TENT
            self.links[(x, y)] = (nx, ny)
            self.empty_cells.discard((x, y))
            self.empty_cells.discard((nx, ny))
            used_tents.add((nx, ny))
            trees_placed += 1

        return trees_placed >= self.num_trees

    def _generate_valid_pairs(self):
        pairs = []
        for x, y in self.empty_cells:
            for nx, ny in self.cached_4[(x, y)]:
                if (nx, ny) in self.empty_cells and not self._has_adjacent_tent(nx, ny):
                    pairs.append(((x, y), (nx, ny)))
        return pairs

    def _has_adjacent_tent(self, x, y):
        for nx, ny in self.cached_8[(x, y)]:
            if self.grid[nx][ny] == _TENT:
                return True
        return False

    def _get_row_col_constraints(self):
        row_constraints = [int(np.sum(row == _TENT)) for row in self.grid]
        col_constraints = [int(np.sum(col == _TENT)) for col in self.grid.T]
        return row_constraints, col_constraints

    def _remove_tents(self):
        for x, y in self.links.values():
            self.grid[x][y] = _EMPTY

    def _validate_grid(self):
        # Проверим, что каждая палатка привязана к ровно одному дереву и наоборот
        for tree, tent in self.links.items():
            if self.grid[tree[0]][tree[1]] != _TREE:
                return False
            if self.grid[tent[0]][tent[1]] != _TENT:
                return False
            # Проверим, что палатка рядом по 4 сторонам с деревом
            if tent not in self.cached_4[tree]:
                return False
            # Проверим, что палатка не касается других палаток по диагонали
            for nx, ny in self.cached_8[tent]:
                if (nx, ny) != tree and self.grid[nx][ny] == _TENT:
                    return False

        # Проверим, что палатка не используется более одного раза
        seen_tents = set()
        for tent in self.links.values():
            if tent in seen_tents:
                return False
            seen_tents.add(tent)

        return True



def print_grid(grid, row_constraints, col_constraints, tent=_TENT):
    size = len(grid)
    print(' ' * 6, end='')
    print(*col_constraints, sep=' ' * 2)
    print('_' * (size * 4))

    for row in range(size):
        print(f'{row_constraints[row]} |', sep='', end=' ')
        for col in range(size):
            if grid[row][col] == tent:
                print("⛺", sep='', end=' ')
            elif grid[row][col] == _TREE:
                print("🌲", sep='', end=' ')
            else:
                print("◻️", sep='', end=' ')
        print()

