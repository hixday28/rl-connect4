import numpy as np
import random
import pickle

class QLearningAgent:
    
    #Класс, который реализует агента, который обучается с подкреплением (Q-Learning).
    #Агент использует Q-таблицу для принятия решений в среде ConnectFour.
    
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, min_epsilon=0.01, epsilon_decay=0.9995):
        
        
        self.alpha = alpha # - Скорость обучения (learning rate)
        self.gamma = gamma # - Коэффициент дисконтирования
        self.epsilon = epsilon # - Начальный коэффициент исследования (exploration rate)
        self.min_epsilon = min_epsilon # - Минимальный коэффициент исследования
        self.epsilon_decay = epsilon_decay # - Коэффициент затухания для эпсилон
        
        self.q_table = {} # Q-таблица хранится в словаре. Ключ - состояние доски, значение - Q-значения для действий

    def get_q_values(self, state_str):
        
        #Получает Q-значения для данного состояния. Если состояния нет в таблице, инициализирует его нулевыми значениями
        
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(7)
        return self.q_table[state_str]

    def choose_action(self, env):
        
        #Выбирает действие (колонку) с использованием эпсилон-жадной стратегии
        
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
        
        max_q = -np.inf
        for move in valid_moves:
            if q_values[move] > max_q:
                max_q = q_values[move]

        best_actions = [move for move, q in valid_q_values.items() if q == max_q]
        
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        
        # Обновляет Q-таблицу на основе полученного опыта
        
        state_str = str(state.tobytes())
        next_state_str = str(next_state.tobytes())
        
        old_value = self.get_q_values(state_str)[action]
        
        if not done:
            next_max = np.max(self.get_q_values(next_state_str))
        else:
            next_max = 0.0

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(7)
        self.q_table[state_str][action] = new_value

        # Вызываем затухание эпсилон после каждого шага обучения
        self.decay_epsilon()

    def decay_epsilon(self):
        #Уменьшает эпсилон для снижения исследования со временем

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filename="q_agent.pkl"):
        # Сохраняет Q-таблицу в файл с помощью pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename="q_agent.pkl"):
        # Загружает Q-таблицу из файла
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print(f"Файл {filename} не найден. Используется пустая Q-таблица.")
            self.q_table = {}
