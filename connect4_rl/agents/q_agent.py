# agents/q_agent.py

import numpy as np
import random
import pickle

class QLearningAgent:
    """
    Класс, реализующий агента, обучающегося с подкреплением (Q-Learning).
    Агент использует Q-таблицу для принятия решений в среде ConnectFour.
    """
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Инициализация агента.
        Аргументы:
            alpha (float): Скорость обучения (learning rate).
            gamma (float): Коэффициент дисконтирования.
            epsilon (float): Коэффициент исследования (exploration rate).
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-таблица хранится в словаре. Ключ - состояние доски, значение - Q-значения для действий.
        self.q_table = {}

    def get_q_values(self, state_str):
        """
        Получает Q-значения для данного состояния. Если состояния нет в таблице,
        инициализирует его нулевыми значениями.
        """
        # .get() возвращает значение для ключа или None, если ключ не найден.
        # or [0.0] * 7 - если .get() вернул None, используется [0.0] * 7
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(7)
        return self.q_table[state_str]

    def choose_action(self, env):
        """
        Выбирает действие (колонку) с использованием эпсилон-жадной стратегии.
        Аргументы:
            env (ConnectFourEnv): Игровая среда.
        Возвращает:
            int: Номер колонки для хода.
        """
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return None # Нет доступных ходов

        # Исследование (Exploration)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)
        
        # Эксплуатация (Exploitation)
        state_str = str(env.get_state().tobytes())
        q_values = self.get_q_values(state_str)
        
        # Выбираем лучшее действие только из числа доступных
        valid_q_values = {move: q_values[move] for move in valid_moves}
        
        # Находим максимальное Q-значение среди доступных ходов
        max_q = -np.inf
        for move in valid_moves:
            if q_values[move] > max_q:
                max_q = q_values[move]

        # Выбираем случайное действие среди всех, имеющих максимальное Q-значение
        best_actions = [move for move, q in valid_q_values.items() if q == max_q]
        
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """
        Обновляет Q-таблицу на основе полученного опыта.
        Формула Q-Learning:
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        """
        state_str = str(state.tobytes())
        next_state_str = str(next_state.tobytes())
        
        old_value = self.get_q_values(state_str)[action]
        
        # Если игра не закончена, ищем максимальное Q-значение для следующего состояния
        if not done:
            next_max = np.max(self.get_q_values(next_state_str))
        else:
            next_max = 0.0 # В конечном состоянии будущая награда равна 0

        # Формула обновления
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        
        # Обновляем Q-таблицу. Убедимся, что состояние существует в таблице.
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(7)
        self.q_table[state_str][action] = new_value

    def save(self, filename="q_agent.pkl"):
        """Сохраняет Q-таблицу в файл с помощью pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename="q_agent.pkl"):
        """Загружает Q-таблицу из файла."""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print(f"Файл {filename} не найден. Используется пустая Q-таблица.")
            self.q_table = {}
