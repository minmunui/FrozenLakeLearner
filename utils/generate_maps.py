import os
from typing import List

from utils.process_IO import create_directory_if_not_exists


# def create_directory_if_not_exists(path):
#     if not os.path.exists(path):
#         os.makedirs(path)


def is_valid(board: List[str]) -> bool:
    max_row_size = len(board)
    max_col_size = len(board[0])
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_row_size or c_new < 0 or c_new >= max_col_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def replace_index_with_char(source: str, index: int, target: str) -> str:
    return source[:index] + target + source[index + 1:]


def make_empty_map(n_col, n_row, start: (int, int), goal: (int, int)):
    """
    This function is used to generate all possible maps of size n_col x n_row
    :param n_col: number of columns
    :param n_row: number of rows
    :param start: start position (row, col)
    :param goal: goal position (row, col)
    :return: empty map
    """
    # 문자열의 배열로 만들기
    board = []
    for i in range(n_row):
        board.append("E" * n_col)

    # 시작점과 도착점 설정
    board[start[1]] = replace_index_with_char(board[start[1]], start[0], "S")
    board[goal[1]] = replace_index_with_char(board[goal[1]], goal[0], "G")
    return board


def is_terminated(board: List[str]) -> bool:
    for row in board:
        if "E" in row:
            return False
    return True


def occupy(board: List[str]) -> List[str]:
    for i in range(len(board)):
        if "E" in board[i]:
            board[i] = replace_index_with_char(board[i], board[i].index("E"), "O")
            break
    return board


def hole(board: List[str]) -> List[str]:
    for i in range(len(board)):
        if "E" in board[i]:
            board[i] = replace_index_with_char(board[i], board[i].index("E"), "H")
            break
    return board


def generate_all_map(n_col: int, n_row: int, start: (int, int) = None, goal: (int, int) = None, dir_path: str = ""):
    """
    This function is used to generate all possible maps of size n_col x n_row
    :param n_col: number of columns
    :param n_row: number of rows
    :param start: position (row, col)
    :param goal: position (row, col)
    :param dir_path: path to the directory to save the maps
    :return: list of all possible maps
    """
    if start is None:
        start = (0, 0)
    if goal is None:
        goal = (n_row - 1, n_col - 1)

    if dir_path == "":
        dir_path = f"maps/generated/all_{n_col}X{n_row}"

    create_directory_if_not_exists(dir_path)

    frontier = [make_empty_map(n_col, n_row, start, goal)]
    map_num = 0

    while frontier:
        current_map = frontier.pop()

        if is_terminated(current_map):
            # if the map is terminated, save the map
            with open(f"{dir_path}/map_{map_num}.txt", "w") as f:
                for row in current_map:
                    f.write(row.replace('O', 'F').replace('E', 'F') + "\n")
            map_num += 1
            continue

        occupied_map, hole_map = occupy(current_map.copy()), hole(current_map.copy())
        frontier.append(occupied_map)
        if is_valid(hole_map):
            frontier.append(hole_map)
