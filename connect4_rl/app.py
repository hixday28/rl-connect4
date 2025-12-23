# Импорты библиотек
import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import plotly.express as px
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func

# Импортируем наши собственные модули
from game.environment import ConnectFourEnv
from agents.q_agent import QLearningAgent
from database.models import (
    Base,
    Agent,
    TrainingSession,
    WinRateLog,
    DB_FILE,
    engine,
)

# Константа для файла с Q-таблицей
Q_TABLE_FILE = "q_agent.pkl"

# Сессия для взаимодействия с БД
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Начальная настройка
if 'env' not in st.session_state:
    st.session_state.env = ConnectFourEnv()
if 'agent' not in st.session_state:
    st.session_state.agent = QLearningAgent()
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'winner' not in st.session_state:
    st.session_state.winner = None
if 'move_history' not in st.session_state:
    st.session_state.move_history = []
if 'current_agent_id' not in st.session_state:
    st.session_state.current_agent_id = None
if 'turn' not in st.session_state:
    st.session_state.turn = 1


# Создаем базу данных при первом запуске, если ее нет
if not os.path.exists(DB_FILE):
    Base.metadata.create_all(bind=engine)

# Вспомогательные функции
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def draw_static_board(board, target_container):
    """Отрисовывает статичное игровое поле в контейнере."""
    styles = """
    <style>
    .board-row { display: flex; flex-direction: row; justify-content: center; }
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
            cell_class = "player1" if player == 1 else "player2" if player == 2 else "empty"
            html_cells.append(f'<div class="cell {cell_class}"></div>')
        html_rows.append(f'<div class="board-row">{"".join(html_cells)}</div>')
    
    board_html = styles + "".join(html_rows)
    target_container.markdown(board_html, unsafe_allow_html=True)


def reset_game():
    st.session_state.env.reset()
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.move_history = []
    st.session_state.turn = 1
    if hasattr(st, 'query_params') and st.query_params:
        st.query_params.clear()


# UI приложения
st.title("Интеллектуальная система автоматизации подготовки ботов для стратегической игры «4 в ряд»")

main_tab, training_tab, game_tab, stats_tab = st.tabs(["Главная", "Обучение", "Игра", "Статистика"])

# ВКЛАДКА "ГЛАВНАЯ"
with main_tab:
    st.header("Добро пожаловать в мир RL!")
    st.write("""
        Это приложение позволяет вам обучать агентов искусственного интеллекта для игры в «4 в ряд» 
        с использованием алгоритмов обучения с подкреплением (Reinforcement Learning).
             
        Вы можете:
        - **Обучать** своих агентов, настраивая различные параметры обучения.
        - **Играть** против обученного агента, чтобы проверить его силу.
        - Просматривать **статистику** обучения и отслеживать прогресс агентов.
    """)

# ВКЛАДКА "ОБУЧЕНИЕ"
with training_tab:
    st.header("Обучение агентов")

    with st.expander("Параметры обучения", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            episodes = st.number_input("Количество игр (эпизодов)", min_value=100, max_value=1000000, value=10000, step=100)
            alpha = st.slider("Скорость обучения (Alpha)", 0.01, 1.0, 0.1, 0.01)
            gamma = st.slider("Дисконт-фактор (Gamma)", 0.8, 0.99, 0.99, 0.01)
        with cols[1]:
            epsilon = st.slider("Начальный Epsilon", 0.1, 1.0, 0.9, 0.05)
            epsilon_decay = st.number_input("Затухание Epsilon", min_value=0.9, max_value=1.0, value=0.9995, step=0.0001, format="%.4f")
    
    demo_mode = st.checkbox("Режим демонстрации (с визуализацией игры)")

    if st.button("Начать обучение"):
        db_gen = get_db()
        db = next(db_gen)
        
        agent_params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon, "epsilon_decay": epsilon_decay}
        existing_agent = db.query(Agent).filter_by(**agent_params).first()

        if not existing_agent:
            new_agent = Agent(**agent_params)
            db.add(new_agent)
            db.commit()
            db.refresh(new_agent)
            agent_id = new_agent.id
        else:
            agent_id = existing_agent.id
        st.session_state.current_agent_id = agent_id

        training_session = TrainingSession(agent_id=agent_id, total_episodes=episodes)
        db.add(training_session)
        db.commit()
        db.refresh(training_session)
        
        agent1 = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, min_epsilon=0.01, epsilon_decay=epsilon_decay)
        agent2 = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, min_epsilon=0.01, epsilon_decay=epsilon_decay)
        env = ConnectFourEnv()

        status_text = st.empty()
        if demo_mode:
            if episodes > 1000:
                episodes = 1000
            board_placeholder = st.empty()
        else:
            status_text.info("Идет обучение агента, пожалуйста, подождите...")
        
        wins_agent1 = 0
        win_rates, chart_placeholder = [], st.empty()
        progress_bar = st.progress(0)
        
        for episode in range(episodes):
            env.reset()
            done = False
            while not done:
                if demo_mode:
                    draw_static_board(env.board, board_placeholder)
                    time.sleep(0.05)
                
                action1 = agent1.choose_action(env)
                if action1 is None: break
                old_state1 = env.get_state()
                next_state, reward1, done, _ = env.step(action1)
                
                if done:
                    reward2 = -10 if reward1 == 10 else 0
                    if reward1 == 10: wins_agent1 += 1
                else:
                    if demo_mode:
                        draw_static_board(env.board, board_placeholder)
                        time.sleep(0.05)
                    
                    action2 = agent2.choose_action(env)
                    if action2 is None: break
                    old_state2 = env.get_state()
                    next_state, reward2, done, _ = env.step(action2)
                    if done and reward2 == 10: reward1 = -10
                    agent2.learn(old_state2, action2, reward2, next_state, done)

                agent1.learn(old_state1, action1, reward1, next_state, done)

            progress_bar.progress((episode + 1) / episodes)
            if (episode + 1) % 100 == 0 or episode == episodes - 1:
                win_rate = (wins_agent1 / (episode + 1)) * 100
                win_rates.append({'episode': episode + 1, 'win_rate': win_rate})
                
                win_rate_log = WinRateLog(session_id=training_session.id, episode_number=episode + 1, win_rate=win_rate)
                db.add(win_rate_log)
                db.commit()

                df_rates = pd.DataFrame(win_rates)
                fig = px.line(df_rates, x='episode', y='win_rate', title="Процент побед Агента 1 (%)")
                chart_placeholder.plotly_chart(fig, use_container_width=True)

        st.success("Обучение завершено!")
        final_win_rate = (wins_agent1 / episodes) * 100
        
        training_session.end_time = datetime.utcnow()
        training_session.final_win_rate = final_win_rate
        db.commit()
        
        agent1.save(Q_TABLE_FILE)
        st.write(f"Финальный процент побед Агента 1: {final_win_rate:.2f}%")
        db.close()

# ВКЛАДКА "ИГРА"
with game_tab:
    st.header("Игра против обученного агента")

    if not os.path.exists(Q_TABLE_FILE):
        st.warning(f"Файл `{Q_TABLE_FILE}` не найден. Сначала обучите агента.")
    else:
        # Загрузка агента, если он еще не в сессии
        if 'q_table' not in st.session_state.agent.q_table or not st.session_state.agent.q_table:
            st.session_state.agent.load(Q_TABLE_FILE)
            st.session_state.agent.epsilon = 0  # Отключаем случайные ходы для игры
            st.success("Агент успешно загружен.")

        # Сообщение о победителе или ничьей
        if st.session_state.game_over:
            winner_map = {1: "Поздравляем, вы победили! 🎉", 2: "Агент победил. Попробуйте еще раз! 🤖", 0: "Ничья! 🤝"}
            st.info(winner_map.get(st.session_state.winner, ""))
        else:
            # Индикатор хода
            if st.session_state.turn == 1:
                st.info("Ваш ход (🔴)")
            else:
                st.warning("Агент думает (🔵)...")
        
        human_action = None
        board_cols = st.columns(st.session_state.env.cols)
        valid_moves = st.session_state.env.get_valid_moves()
        
        for i in range(st.session_state.env.cols):
            with board_cols[i]:
                # Отрисовка ячеек столбца
                for r in range(st.session_state.env.rows):
                    player = st.session_state.env.board[r, i]
                    if player == 1:
                        emoji = "🔴"
                    elif player == 2:
                        emoji = "🔵"
                    else:
                        emoji = "⚪️"
                    st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
                
                # Кнопка для хода
                is_disabled = (i not in valid_moves) or st.session_state.game_over
                if st.button("⬇️", key=f"btn_{i}", disabled=is_disabled, use_container_width=True):
                    human_action = i

        # Логика обработки ходов
        if human_action is not None:
            # Ход человека
            st.session_state.move_history.append((1, human_action))
            _, _, human_done, _ = st.session_state.env.step(human_action)
            st.session_state.turn = 2 # Переключаем ход на агента
            
            if human_done:
                st.session_state.game_over = True
                st.session_state.winner = 1 if st.session_state.env.check_win(1) else 0
            else:
                # Ход агента
                action = st.session_state.agent.choose_action(st.session_state.env)
                if action is not None:
                    st.session_state.move_history.append((2, action))
                    _, _, agent_done, _ = st.session_state.env.step(action)
                    if agent_done:
                        st.session_state.game_over = True
                        st.session_state.winner = 2 if st.session_state.env.check_win(2) else 0
                st.session_state.turn = 1 # Возвращаем ход человеку
            
            st.rerun()

        if st.session_state.game_over:
            if st.button("Новая игра", use_container_width=True):
                reset_game()
                st.rerun()


# ВКЛАДКА "СТАТИСТИКА"
with stats_tab:
    st.header("Статистика и история игр")
    
    db_gen = get_db()
    db = next(db_gen)
    
    st.subheader("Реестр уникальных агентов")
    avg_win_rates = db.query(
        TrainingSession.agent_id,
        func.avg(TrainingSession.final_win_rate).label('avg_win_rate'),
        func.count(TrainingSession.id).label('session_count')
    ).group_by(TrainingSession.agent_id).subquery()
    
    agents_query = db.query(Agent, avg_win_rates.c.avg_win_rate, avg_win_rates.c.session_count).outerjoin(
        avg_win_rates, Agent.id == avg_win_rates.c.agent_id
    )
    agents_data = [{
        "ID Агента": agent.id, "Alpha": agent.alpha, "Gamma": agent.gamma,
        "Epsilon": agent.epsilon, "Decay": agent.epsilon_decay,
        "Средний WinRate (%)": f"{avg_win_rate:.2f}" if avg_win_rate else "N/A",
        "Кол-во сессий": session_count if session_count else 0
    } for agent, avg_win_rate, session_count in agents_query.all()]

    if not agents_data:
        st.info("Нет данных по агентам. Проведите обучение.")
    else:
        st.dataframe(pd.DataFrame(agents_data))

    st.subheader("История сессий обучения")
    sessions = db.query(TrainingSession).order_by(TrainingSession.start_time.desc()).all()
    
    if not sessions:
        st.info("Пока не было проведено ни одной сессии обучения.")
    else:
        sessions_data = [{
            "ID Сессии": s.id, "ID Агента": s.agent_id,
            "Время начала": s.start_time.strftime("%Y-%m-%d %H:%M"),
            "Длительность (сек)": f"{(s.end_time - s.start_time).total_seconds():.2f}" if s.end_time else 0,
            "Кол-во эпизодов": s.total_episodes,
            "Финальный WinRate (%)": f"{s.final_win_rate:.2f}" if s.final_win_rate is not None else "N/A",
        } for s in sessions]
        sessions_df = pd.DataFrame(sessions_data)
        st.dataframe(sessions_df)

        session_to_view = st.selectbox("Выберите сессию для просмотра кривой обучения", options=sessions_df["ID Сессии"])
        if session_to_view:
            with st.expander(f"Кривая обучения для сессии #{session_to_view}"):
                logs = db.query(WinRateLog).filter(WinRateLog.session_id == session_to_view).order_by(WinRateLog.episode_number).all()
                if not logs:
                    st.warning("Нет данных для построения графика для этой сессии.")
                else:
                    log_df = pd.DataFrame([{"Эпизод": log.episode_number, "Win Rate (%)": log.win_rate} for log in logs])
                    fig = px.line(log_df, x="Эпизод", y="Win Rate (%)", title=f"Кривая обучения для сессии #{session_to_view}")
                    st.plotly_chart(fig, use_container_width=True)
    db.close()
