import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os

# Путь к файлу базы данных
DB_FILE = "training_stats.db"

# URL для подключения. 'sqlite:///' означает, что файл будет в корне проекта.
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Создаем движок SQLAlchemy
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Сессия для взаимодействия с БД
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#Базовый класс для всех моделей
Base = declarative_base()

#Модель данных
class TrainingSession(Base):
    """
    Модель SQLAlchemy для хранения результатов сессии обучения.
    """
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    episodes_run = Column(Integer, nullable=False)
    win_rate_agent_1 = Column(Float, nullable=False)

#Функции для работы с БД

def init_db():
    
    #Создаем все таблицы, которые наследуются от Base
    Base.metadata.create_all(bind=engine)

def save_training_result(episodes, win_rate):
    
    #Сохраняет результат одной сессии обучения в базу данных.
    
    db = SessionLocal()
    try:
        session_result = TrainingSession(
            episodes_run=episodes,
            win_rate_agent_1=win_rate
        )
        db.add(session_result)
        db.commit()
    finally:
        db.close()

def get_all_results():
    
    #Извлекает все результаты обучения из базы данных

    db = SessionLocal()
    try:
        # Выполняем запрос и сразу читаем результат в pandas DataFrame
        query = db.query(TrainingSession).statement
        df = pd.read_sql(query, db.bind)
        return df
    finally:
        db.close()

#Проверяем, существует ли файл БД. Если нет то создаем и инициализируем
if not os.path.exists(DB_FILE):
    print(f"База данных {DB_FILE} не найдена. Создание новой...")
    init_db()
    print("База данных успешно создана.")
