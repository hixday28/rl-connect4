# Импорты библиотек
import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import plotly.express as px

# Импортируем наши собственные модули
from game.environment import ConnectFourEnv
from agents.q_agent import QLearningAgent
from database.models import init_db, save_training_result, get_all_results

# Константа для файла с Q-таблицей
Q_TABLE_FILE = "q_agent.pkl"

# Начальная настройка

# Используем session_state, чтобы сохранить состояние игры между действиями пользователя
if 'env' not in st.session_state:
    st.session_state.env = ConnectFourEnv()
if 'agent' not in st.session_state:
    st.session_state.agent = QLearningAgent()
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'winner' not in st.session_state:
    st.session_state.winner = None

# Создаем базу данных при первом запуске
init_db()


#Вспомогательные функции:

# Функция для отрисовки доски
def draw_board(board, target_container=None):
    if target_container is None:
        target_container = st.container()

    # Сначала чистим контейнер
    target_container.empty()

    # Стили для ячеек доски
    styles = """
    <style>
    .board-row { display: flex; flex-direction: row; justify-content: center; }
    .cell { width: 50px; height: 50px; border-radius: 50%; display: inline-block; margin: 3px; border: 2px solid #ccc; }
    .player1 { background-color: #FF4B4B; }
    .player2 { background-color: #4B7BFF; }
    .empty { background-color: #FFFFFF; }
    </style>
    """

    # Генерируем HTML для доски
    html_rows = []
    for r in range(board.shape[0]):
        html_cells = []
        for c in range(board.shape[1]):
            player = board[r, c]
            if player == 1: cell_class = "player1"
            elif player == 2: cell_class = "player2"
            else: cell_class = "empty"
            html_cells.append(f'<div class="cell {cell_class}"></div>')
        
        html_rows.append(f'<div class="board-row">{"".join(html_cells)}</div>')

    full_board_html = "".join(html_rows)

    target_container.markdown(styles + full_board_html, unsafe_allow_html=True)

# Функция для сброса игры
def reset_game():
    st.session_state.env.reset()
    st.session_state.game_over = False
    st.session_state.winner = None

#UI приложения:

st.title("Интеллектуальная система автоматизации подготовки ботов для стратегической игры «4 в ряд»")

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["Обучение", "Игра", "Статистика"])

#ВКЛАДКА "ОБУЧЕНИЕ"
with tab1:
    st.header("Обучение агентов")

    # Параметры для обучения в боковой панели
    st.sidebar.title("Параметры обучения")
    episodes = st.sidebar.number_input("Количество игр (эпизодов)", min_value=100, max_value=1000000, value=10000, step=100)
    alpha = st.sidebar.slider("Скорость обучения (Alpha)", 0.01, 1.0, 0.1, 0.01)
    gamma = st.sidebar.slider("Дисконт-фактор (Gamma)", 0.8, 0.99, 0.99, 0.01)
    epsilon = st.sidebar.slider("Начальный Epsilon", 0.1, 1.0, 0.9, 0.05)
    min_epsilon = st.sidebar.number_input("Минимальный Epsilon", min_value=0.0, max_value=0.2, value=0.01, step=0.01, format="%.2f")
    epsilon_decay = st.sidebar.number_input("Затухание Epsilon", min_value=0.9, max_value=1.0, value=0.9995, step=0.0001, format="%.4f")
    
    demo_mode = st.checkbox("Режим демонстрации (с визуализацией игры)")

    # Кнопка запуска обучения
    if st.button("Начать обучение"):
        # Создаем двух агентов для игры друг против друга
        agent1 = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay)
        agent2 = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay)
        env = ConnectFourEnv()
        
        # Элементы для отображения прогресса
        status_text = st.empty()
        if demo_mode:
            if episodes > 1000:
                st.warning("В режиме демонстрации количество эпизодов ограничено до 1000.")
                episodes = 1000
            board_placeholder = st.empty()
        else:
            status_text.info("Идет обучение агента, пожалуйста, подождите...")

        wins_agent1 = 0
        win_rates = []
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()
        
        st.write(f"Начинаем обучение на {episodes} эпизодах...")

        # Основной цикл обучения
        for episode in range(episodes):
            env.reset()
            state = env.get_state()
            done = False
            
            while not done:
                if demo_mode: # Показываем доску в демо-режиме
                    draw_board(env.board, target_container=board_placeholder)
                    time.sleep(0.05)

                # Ход первого агента
                action1 = agent1.choose_action(env)
                if action1 is None: break 
                
                old_state1 = env.get_state()
                next_state, reward1, done, info = env.step(action1)
                
                if done: # ксли игра закончилась на ходе первого
                    if reward1 == 10: wins_agent1 += 1; reward2 = -10
                    elif 'error' in info: reward2 = 10 
                    else: reward2 = 0
                else:
                    if demo_mode:
                        draw_board(env.board, target_container=board_placeholder)
                        time.sleep(0.05)
                        
                    # ход второго агента
                    action2 = agent2.choose_action(env)
                    if action2 is None: break
                    
                    old_state2 = env.get_state()
                    next_state, reward2, done, info = env.step(action2)

                    if done and reward2 == 10: reward1 = -10 # Если второй выиграл
                    
                    # обучаем второго агента
                    agent2.learn(old_state2, action2, reward2, next_state, done)

                # обучаем первого агента
                agent1.learn(old_state1, action1, reward1, next_state, done)
                state = next_state

            # Обновляем прогресс и график
            progress_bar.progress((episode + 1) / episodes)
            if (episode + 1) % 100 == 0 or episode == episodes - 1:
                win_rate = (wins_agent1 / (episode + 1)) * 100
                win_rates.append({'episode': episode + 1, 'win_rate': win_rate})
                df_rates = pd.DataFrame(win_rates)
                fig = px.line(df_rates, x='episode', y='win_rate', title="Процент побед Агента 1 (%)")
                chart_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.success("Обучение завершено!")
        status_text.empty()
        
        # Сохраняем результаты в БД и файл
        final_win_rate = (wins_agent1 / episodes) * 100
        agent1.save(Q_TABLE_FILE)
        save_training_result(episodes, final_win_rate)
        
        st.write(f"Финальный процент побед Агента 1: {final_win_rate:.2f}%")
        st.write(f"Q-таблица победившего агента сохранена в `{Q_TABLE_FILE}`.")


#ВКЛАДКА "ИГРА"
with tab2:
    st.header("Игра против обученного агента")
    
    # Проверяем, есть ли сохраненный агент
    if not os.path.exists(Q_TABLE_FILE):
        st.warning(f"Файл с обученным агентом `{Q_TABLE_FILE}` не найден. Сначала обучите агента на вкладке 'Обучение'.")
    else:
        # Загружаем Q-таблицу агента
        if 'q_table' not in st.session_state.agent.q_table or not st.session_state.agent.q_table:
            st.session_state.agent.load(Q_TABLE_FILE)
            st.session_state.agent.epsilon = 0 # Отключаем исследование для игры
            st.success(f"Агент успешно загружен из `{Q_TABLE_FILE}`.")

        # Показываем сообщение о результате игры
        if st.session_state.game_over:
            if st.session_state.winner == 1: st.success("Поздравляем, вы победили! 🎉")
            elif st.session_state.winner == 2: st.error("Агент победил. Попробуйте еще раз! 🤖")
            else: st.info("Ничья! 🤝")

        # Рисуем поле и кнопки для хода
        valid_moves = st.session_state.env.get_valid_moves()
        cols = st.columns(st.session_state.env.cols)
        human_action = None

        for i in range(st.session_state.env.cols):
            with cols[i]:
                # Ячейки колонки
                for r in range(st.session_state.env.rows):
                    player = st.session_state.env.board[r, i]
                    icon = "⚪️"
                    if player == 1: icon = "🔴"
                    elif player == 2: icon = "🔵"
                    st.markdown(f"<p style='text-align: center; font-size: 28px; height: 40px;'>{icon}</p>", unsafe_allow_html=True)
                
                # Кнопка для хода в колонку
                is_disabled = (i not in valid_moves) or st.session_state.game_over or (st.session_state.env.current_player != 1)
                if st.button("⬇️", key=f"btn_{i}", disabled=is_disabled, use_container_width=True):
                    human_action = i

        # Логика игры после нажатия кнопки
        if human_action is not None:
            # Ход человека
            _, _, human_done, _ = st.session_state.env.step(human_action)
            if human_done:
                st.session_state.game_over = True
                st.session_state.winner = 1 if st.session_state.env.check_win(1) else 0
                st.rerun() # Перезапускаем скрипт для обновления UI
            else:
                # Ход агента
                action = st.session_state.agent.choose_action(st.session_state.env)
                if action is not None:
                    _, _, agent_done, _ = st.session_state.env.step(action)
                    if agent_done:
                        st.session_state.game_over = True
                        st.session_state.winner = 2 if st.session_state.env.check_win(2) else 0
                st.rerun()

        # Показываем кнопку "Новая игра" после окончания
        if st.session_state.game_over:
            if st.button("Новая игра", use_container_width=True):
                reset_game()
                st.rerun()

#"СТАТИСТИКА"
with tab3:
    st.header("История и статистика обучений")
    
    # Получаем данные из БД
    results_df = get_all_results()
    
    if results_df.empty:
        st.info("Пока нет данных для отображения. Проведите хотя бы одну сессию обучения.")
    else:
        # Показываем таблицу с результатами
        st.table(results_df)
        
        # И график изменения процента побед
        fig = px.line(results_df, x='timestamp', y='win_rate_agent_1', 
                      title='Процент побед Агента 1 по сессиям обучения',
                      labels={'timestamp': 'Дата и время', 'win_rate_agent_1': 'Процент побед (%)'},
                      markers=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})