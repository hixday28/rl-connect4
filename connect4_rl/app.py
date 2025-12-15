# app.py

import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import plotly.express as px

# Импортируем наши классы и функции
from game.environment import ConnectFourEnv
from agents.q_agent import QLearningAgent
from database.models import init_db, save_training_result, get_all_results

# --- Константы и конфигурация ---
Q_TABLE_FILE = "q_agent.pkl"

# --- Инициализация ---
# Инициализируем сессию Streamlit
if 'env' not in st.session_state:
    st.session_state.env = ConnectFourEnv()
if 'agent' not in st.session_state:
    st.session_state.agent = QLearningAgent()
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'winner' not in st.session_state:
    st.session_state.winner = None

# Убедимся, что БД инициализирована
init_db()


# --- Вспомогательные функции ---

def draw_board(board, target_container=None):
    """Отрисовывает игровую доску в Streamlit с помощью одного HTML-блока."""
    # Если контейнер не передан, создаем новый st.container()
    if target_container is None:
        target_container = st.container()

    # Очищаем контейнер перед отрисовкой
    target_container.empty()

    styles = """
    <style>
    .board-row {
        display: flex;
        flex-direction: row;
        justify-content: center;
    }
    .cell {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: inline-block;
        margin: 3px;
        border: 2px solid #ccc;
    }
    .player1 { background-color: #FF4B4B; }
    .player2 { background-color: #4B7BFF; }
    .empty { background-color: #FFFFFF; }
    </style>
    """

    html_rows = []
    for r in range(board.shape[0]):
        html_cells = []
        for c in range(board.shape[1]):
            player = board[r, c]
            if player == 1:
                cell_class = "player1"
            elif player == 2:
                cell_class = "player2"
            else:
                cell_class = "empty"
            html_cells.append(f'<div class="cell {cell_class}"></div>')
        
        html_rows.append(f'<div class="board-row">{" ".join(html_cells)}</div>')

    full_board_html = "".join(html_rows)

    target_container.markdown(styles + full_board_html, unsafe_allow_html=True)

def reset_game():
    """Сбрасывает состояние игры."""
    st.session_state.env.reset()
    st.session_state.game_over = False
    st.session_state.winner = None

# --- Основной интерфейс ---

st.title("🤖 Интеллектуальная система обучения 'Connect 4'")

tab1, tab2, tab3 = st.tabs(["Обучение", "Игра", "Статистика"])

# --- ВКЛАДКА "ОБУЧЕНИЕ" ---
with tab1:
    st.header("Обучение агентов")

    st.sidebar.title("Параметры обучения")
    episodes = st.sidebar.number_input("Количество игр (эпизодов)", min_value=100, max_value=1000000, value=10000, step=100)
    alpha = st.sidebar.slider("Скорость обучения (Alpha)", 0.01, 1.0, 0.1, 0.01)
    gamma = st.sidebar.slider("Дисконт-фактор (Gamma)", 0.8, 0.99, 0.99, 0.01)
    epsilon = st.sidebar.slider("Коэффициент исследования (Epsilon)", 0.05, 0.5, 0.1, 0.05)
    
    demo_mode = st.checkbox("Режим демонстрации (с визуализацией игры)")

    if st.button("Начать обучение"):
        agent1 = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
        agent2 = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
        env = ConnectFourEnv()
        
        status_text = st.empty() # Плейсхолдер для статуса обучения
        
        # Настройка в зависимости от режима
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

        for episode in range(episodes):
            env.reset()
            state = env.get_state()
            done = False
            
            while not done:
                # Отрисовка доски в демо-режиме
                if demo_mode:
                    draw_board(env.board, target_container=board_placeholder)
                    time.sleep(0.05) # Небольшая задержка для наглядности

                # Ход Агента 1
                action1 = agent1.choose_action(env)
                if action1 is None: break 
                
                old_state1 = env.get_state()
                next_state, reward1, done, info = env.step(action1)
                
                if done:
                    if reward1 == 1: # Агент 1 выиграл
                        wins_agent1 += 1
                        reward2 = -1 # Агент 2 проиграл
                    elif 'error' in info: # Неверный ход
                        reward2 = 1 # Вознаграждаем второго агента за ошибку первого
                    else: # Ничья
                        reward2 = 0
                else:
                    # Отрисовка доски в демо-режиме после хода агента 1
                    if demo_mode:
                        draw_board(env.board, target_container=board_placeholder)
                        time.sleep(0.05)
                        
                    # Ход Агента 2
                    action2 = agent2.choose_action(env)
                    if action2 is None: break
                    
                    old_state2 = env.get_state()
                    next_state, reward2, done, info = env.step(action2)

                    if done and reward2 == 1: # Агент 2 выиграл
                        reward1 = -1 # Агент 1 проиграл
                    else: # Ничья или обычный ход
                        reward1 = 0
                    
                    # Обучаем агента 2
                    agent2.learn(old_state2, action2, reward2, next_state, done)

                # Обучаем агента 1
                agent1.learn(old_state1, action1, reward1, next_state, done)
                state = next_state

            # Обновление UI
            progress_bar.progress((episode + 1) / episodes)
            if (episode + 1) % 100 == 0 or episode == episodes - 1:
                win_rate = (wins_agent1 / (episode + 1)) * 100
                win_rates.append({'episode': episode + 1, 'win_rate': win_rate})
                df_rates = pd.DataFrame(win_rates)
                fig = px.line(df_rates, x='episode', y='win_rate', title="Win Rate Агента 1 (%)")
                chart_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.success("Обучение завершено!")
        status_text.empty() # Очищаем сообщение о статусе после завершения обучения
        
        # Сохранение результатов
        final_win_rate = (wins_agent1 / episodes) * 100
        agent1.save(Q_TABLE_FILE)
        save_training_result(episodes, final_win_rate)
        
        st.write(f"Финальный Win Rate Агента 1: {final_win_rate:.2f}%")
        st.write(f"Q-таблица победившего агента сохранена в `{Q_TABLE_FILE}`.")


# --- ВКЛАДКА "ИГРА" ---
with tab2:
    st.header("Игра против обученного агента")
    
    if not os.path.exists(Q_TABLE_FILE):
        st.warning(f"Файл с обученным агентом `{Q_TABLE_FILE}` не найден. Сначала обучите агента на вкладке 'Обучение'.")
    else:
        # Загружаем агента, если он еще не в сессии
        if not st.session_state.agent.q_table:
            st.session_state.agent.load(Q_TABLE_FILE)
            st.session_state.agent.epsilon = 0 # В режиме игры агент не исследует
            st.success(f"Агент успешно загружен из `{Q_TABLE_FILE}`.")

        # Отображение доски
        draw_board(st.session_state.env.board)

        if st.session_state.game_over:
            if st.session_state.winner == 1:
                st.success("Поздравляем, вы победили! 🎉")
            elif st.session_state.winner == 2:
                st.error("Агент победил. Попробуйте еще раз! 🤖")
            else:
                st.info("Ничья! 🤝")
            
            if st.button("Новая игра"):
                reset_game()
                st.rerun()

        else:
            # Ход человека (Игрок 1)
            if st.session_state.env.current_player == 1:
                cols = st.columns(7)
                valid_moves = st.session_state.env.get_valid_moves()

                for i in range(7):
                    with cols[i]:
                        if st.button(f"Ход {i+1}", disabled=(i not in valid_moves)):
                            # Ход человека
                            _, reward, done, _ = st.session_state.env.step(i)
                            
                            if done:
                                st.session_state.game_over = True
                                st.session_state.winner = 1 if reward == 1 else 0
                                st.rerun()

                            # Ход агента (Игрок 2)
                            if not st.session_state.game_over:
                                action = st.session_state.agent.choose_action(st.session_state.env)
                                if action is not None:
                                    _, reward, done, _ = st.session_state.env.step(action)
                                    if done:
                                        st.session_state.game_over = True
                                        st.session_state.winner = 2 if reward == 1 else 0
                                st.rerun()

# --- ВКЛАДКА "СТАТИСТИКА" ---
with tab3:
    st.header("История и статистика обучений")
    
    results_df = get_all_results()
    
    if results_df.empty:
        st.info("Пока нет данных для отображения. Проведите хотя бы одну сессию обучения.")
    else:
        st.dataframe(results_df)
        
        fig = px.line(results_df, x='timestamp', y='win_rate_agent_1', 
                      title='Win Rate Агента 1 по сессиям обучения',
                      labels={'timestamp': 'Дата и время', 'win_rate_agent_1': 'Win Rate (%)'},
                      markers=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})