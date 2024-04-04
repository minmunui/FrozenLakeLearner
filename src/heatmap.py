import numpy as np
import matplotlib.pyplot as plt
import os

from heatmapInput import heatmap_input


# 폴더에서 모든 파일을 읽어서 F 발생 횟수를 누적하는 함수
def accumulate_F_counts_in_folder(folder_path):
    # 폴더 내 파일들의 목록을 가져오기, 현재 폴더 경로를 이용하여 파일 목록을 가져옴
    file_list = os.listdir(folder_path)

    # 초기 누적 배열 생성
    max_rows, max_cols = 0, 0
    for file_name in file_list:
        if file_name.endswith('.txt'):  # 텍스트 파일인 경우에만 처리
            with open(os.path.join(folder_path, file_name), 'r') as file:
                lines = file.readlines()
                max_rows = max(max_rows, len(lines))
                max_cols = max(max_cols, len(lines[0].strip()))

    cumulative_counts = np.zeros((max_rows, max_cols))

    # 각 파일을 읽어서 F 발생 횟수를 누적
    for file_name in file_list:
        if file_name.endswith('.txt'):  # 텍스트 파일인 경우에만 처리
            with open(os.path.join(folder_path, file_name), 'r') as file:
                lines = file.readlines()
                rows = len(lines)
                cols = len(lines[0].strip())

                for i in range(rows):
                    for j in range(cols):
                        if lines[i][j] == 'F':
                            cumulative_counts[i][j] += 1

    return cumulative_counts


# 히트맵을 그리는 함수
def draw_heatmap(counts):
    plt.imshow(counts, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


def heatmap_command():
    # 파일이 들어있는 폴더 경로
    folder_path = heatmap_input()['folder_path']

    # 모든 파일을 읽어서 F 발생 횟수를 누적하여 히트맵 그리기
    cumulative_counts = accumulate_F_counts_in_folder(folder_path)
    print("cumulative_counts : \n", cumulative_counts)
    draw_heatmap(cumulative_counts)
