# game/environment.py

import numpy as np

class ConnectFourEnv:
    """
    Класс, представляющий среду для игры "4 в ряд" (Connect Four).
    Среда предоставляет стандартный интерфейс для RL-агентов.
    """
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # Начинает игрок 1

    def reset(self):
        """
        Сбрасывает доску к начальному состоянию.
        Возвращает:
            np.array: Начальное состояние доски.
        """
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.board

    def get_valid_moves(self):
        """
        Возвращает список колонок, в которые можно сделать ход.
        """
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def step(self, col):
        """
        Выполняет ход в указанной колонке.
        Аргументы:
            col (int): Номер колонки (0-6).
        Возвращает:
            tuple: (state, reward, done, info)
                - state (np.array): Новое состояние доски.
                - reward (float): Награда за ход.
                - done (bool): True, если игра завершена.
                - info (dict): Дополнительная информация.
        """
        # Проверка, является ли ход допустимым
        if col not in self.get_valid_moves():
            # Недопустимый ход, наказываем агента
            return self.board, -1, True, {'error': 'Invalid move'}

        # Находим первую свободную строку в колонке
        row = -1
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                row = r
                break
        
        self.board[row][col] = self.current_player

        # Проверка на победу
        if self.check_win(self.current_player):
            reward = 1.0  # Победа
            done = True
        # Проверка на ничью
        elif len(self.get_valid_moves()) == 0:
            reward = 0.0  # Ничья
            done = True
        else:
            reward = 0.0  # Обычный ход
            done = False

        # Смена игрока
        self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
        
        return self.board, reward, done, {}

    def check_win(self, player):
        """
        Проверяет, выиграл ли указанный игрок.
        Аргументы:
            player (int): Игрок (1 или 2).
        Возвращает:
            bool: True, если игрок выиграл.
        """
        # Горизонтальная проверка
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r][c+i] == player for i in range(4)):
                    return True

        # Вертикальная проверка
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if all(self.board[r+i][c] == player for i in range(4)):
                    return True

        # Положительная диагональ (/)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r-i][c+i] == player for i in range(4)):
                    return True

        # Отрицательная диагональ (\)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(self.board[r+i][c+i] == player for i in range(4)):
                    return True
        
        return False
    
    def get_state(self):
        """Возвращает текущее состояние доски."""
        return self.board
