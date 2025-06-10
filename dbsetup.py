from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, DateTime,
    ForeignKey, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

Base = declarative_base()

class Role(Base):
    __tablename__ = 'roles'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(200))
    users = relationship('User', back_populates='role')

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    role_id = Column(Integer, ForeignKey('roles.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    role = relationship('Role', back_populates='users')

class Ticker(Base):
    __tablename__ = 'tickers'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    market_data = relationship('MarketData', back_populates='ticker')
    predictions = relationship('Prediction', back_populates='ticker')

class MarketData(Base):
    __tablename__ = 'market_data'
    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey('tickers.id'), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    ticker = relationship('Ticker', back_populates='market_data')
    indicators = relationship('Indicator', back_populates='market_data')

class Indicator(Base):
    __tablename__ = 'indicators'
    id = Column(Integer, primary_key=True)
    market_data_id = Column(Integer, ForeignKey('market_data.id'), nullable=False)
    name = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    market_data = relationship('MarketData', back_populates='indicators')

class ModelInfo(Base):
    __tablename__ = 'model_info'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    parameters = Column(JSON)
    trained_at = Column(DateTime, default=datetime.datetime.utcnow)
    mae = Column(Float)
    rmse = Column(Float)
    predictions = relationship('Prediction', back_populates='model')

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('model_info.id'), nullable=False)
    ticker_id = Column(Integer, ForeignKey('tickers.id'), nullable=False)
    date = Column(Date, nullable=False)
    predicted_close = Column(Float, nullable=False)
    model = relationship('ModelInfo', back_populates='predictions')
    ticker = relationship('Ticker', back_populates='predictions')

class LogEntry(Base):
    __tablename__ = 'log_entries'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    level = Column(String(20), nullable=False)
    message = Column(String(500))

def create_database(uri='sqlite:///analytics.db'):
    engine = create_engine(uri, echo=True)
    Base.metadata.create_all(engine)
    return engine

def seed_database(engine):
    Session = sessionmaker(bind=engine)
    session = Session()

    # Roles and a test user
    admin_role = Role(name='admin', description='Administrator with all permissions')
    user_role = Role(name='user', description='Standard user')
    session.add_all([admin_role, user_role])
    session.commit()

    alice = User(username='alice', email='alice@example.com', role=admin_role)
    session.add(alice)
    session.commit()

    gold = Ticker(symbol='XAUUSD', name='Gold Spot Price')
    session.add(gold)
    session.commit()

    today = datetime.date.today()
    sample_prices = [1950.0, 1955.5, 1960.2, 1958.0, 1962.8]
    for i, price in enumerate(sample_prices):
        md = MarketData(
            ticker=gold,
            date=today - datetime.timedelta(days=len(sample_prices) - i),
            open=price - 5,
            high=price + 10,
            low=price - 10,
            close=price,
            volume=1000 + i * 100
        )
        session.add(md)
    session.commit()

    last_md = session.query(MarketData).filter_by(ticker=gold).order_by(MarketData.date.desc()).first()
    sma = Indicator(market_data=last_md, name='SMA_5', value=sum(sample_prices)/len(sample_prices))
    session.add(sma)
    session.commit()

    model = ModelInfo(
        name='LinearRegression_5day',
        parameters={'order': 1},
        mae=2.5,
        rmse=3.1
    )
    session.add(model)
    session.commit()

    prediction = Prediction(
        model=model,
        ticker=gold,
        date=today + datetime.timedelta(days=1),
        predicted_close=1965.0
    )
    session.add(prediction)
    session.commit()

    log = LogEntry(level='INFO', message='Database seeded with gold forecasting data')
    session.add(log)
    session.commit()

    print('Seeding completed.')

if __name__ == '__main__':
    engine = create_database()
    seed_database(engine)
    print('Database created and seeded successfully.')