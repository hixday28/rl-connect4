# connect4_rl/database/models.py

import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# Путь к файлу базы данных
DB_FILE = "training_stats.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Создаем движок SQLAlchemy
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Сессия для взаимодействия с БД
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для всех моделей
Base = declarative_base()

# 1. Модель Agent (Агенты)
class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    alpha = Column(Float, nullable=False)
    gamma = Column(Float, nullable=False)
    epsilon = Column(Float, nullable=False)
    epsilon_decay = Column(Float, nullable=False)

    training_sessions = relationship("TrainingSession", back_populates="agent")

# 2. Модель TrainingSession (Сессии обучения)
class TrainingSession(Base):
    __tablename__ = "training_sessions"
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_episodes = Column(Integer, nullable=False)
    final_win_rate = Column(Float, nullable=True)

    agent = relationship("Agent", back_populates="training_sessions")
    win_rate_logs = relationship("WinRateLog", back_populates="session", cascade="all, delete-orphan")

# 3. Модель WinRateLog (Лог процента побед)
class WinRateLog(Base):
    __tablename__ = "win_rate_logs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("training_sessions.id"), nullable=False)
    episode_number = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)

    session = relationship("TrainingSession", back_populates="win_rate_logs")


# Функции для работы с БД
def init_db():
    """Создает все таблицы, которые наследуются от Base."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Возвращает сессию базы данных."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Проверяем, существует ли файл БД. Если нет, то создаем и инициализируем
if not os.path.exists(DB_FILE):
    print(f"База данных {DB_FILE} не найдена. Создание новой...")
    init_db()
    print("База данных успешно создана.")