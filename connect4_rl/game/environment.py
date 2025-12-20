# game/environment.py

import numpy as np

class ConnectFourEnv:
    
    #Класс, представляющий среду для игры
    
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # Начинает игрок 1

    def reset(self):
        
        #Сбрасывает доску к начальному состоянию.
        
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.board

    def get_valid_moves(self):
        # Возвращает список колонок, в которые можно сделать ход.
        
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def _count_sequences(self, player, length):
        
        #Подсчитывает количество последовательностей (угроз) определенной длины для игрока
        
        count = 0
        # Проверка по всем направлениям
        for r in range(self.rows):
            for c in range(self.cols):
                # Горизонталь
                if c <= self.cols - 4:
                    window = self.board[r, c:c+4]
                    if np.count_nonzero(window == player) == length and np.count_nonzero(window == 0) == 4 - length:
                        count += 1
                #вертикаль
                if r <= self.rows - 4:
                    window = self.board[r:r+4, c]
                    if np.count_nonzero(window == player) == length and np.count_nonzero(window == 0) == 4 - length:
                        count += 1
                # положительная диагональ (/)
                if r >= 3 and c <= self.cols - 4:
                    window = np.array([self.board[r-i, c+i] for i in range(4)])
                    if np.count_nonzero(window == player) == length and np.count_nonzero(window == 0) == 4 - length:
                        count += 1
                # Отрицательная диагональ (\)
                if r <= self.rows - 4 and c <= self.cols - 4:
                    window = np.array([self.board[r+i, c+i] for i in range(4)])
                    if np.count_nonzero(window == player) == length and np.count_nonzero(window == 0) == 4 - length:
                        count += 1
        return count

    def step(self, col):
        
        # Выполняет ход в указанной колонке.

        mover = self.current_player
        
        #проверка является ли ход допустимым
        if col not in self.get_valid_moves():
            return self.board, -10.0, True, {'error': 'Invalid move'}

        #находим первую свободную строку в колонке
        row = -1
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                row = r
                break
        
        self.board[row][col] = mover

        # Проверка на победу
        if self.check_win(mover):
            reward = 10.0
            done = True
        # Проверка на ничью
        elif len(self.get_valid_moves()) == 0:
            reward = 0.0
            done = True
        else:
            # Промежуточная награда (Reward Shaping)
            threes = self._count_sequences(mover, 3)
            twos = self._count_sequences(mover, 2)
            reward = 0.1 * threes + 0.05 * twos
            done = False

        # Смена игрока
        self.current_player = 3 - mover
        
        return self.board, reward, done, {}

    def check_win(self, player):
        
        #Проверяет, выиграл ли указанный игрок.
        
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
        # Возвращает текущее состояние доски
        return self.board
