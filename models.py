from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import relationship
from db import Base

class Role(Base):
    __tablename__ = 'roles'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    users = relationship('User', back_populates='role')

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    role_id = Column(Integer, ForeignKey('roles.id'))
    role = relationship('Role', back_populates='users')


class Ticker(Base):
    __tablename__ = 'tickers'
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    market_data = relationship('MarketData', back_populates='ticker')

class MarketData(Base):
    __tablename__ = 'market_data'
    id = Column(Integer, primary_key=True, index=True)
    ticker_id = Column(Integer, ForeignKey('tickers.id'))
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    ticker = relationship('Ticker', back_populates='market_data')
    indicators = relationship('Indicator', back_populates='market_data')

class Indicator(Base):
    __tablename__ = 'indicators'
    id = Column(Integer, primary_key=True, index=True)
    market_data_id = Column(Integer, ForeignKey('market_data.id'))
    name = Column(String, index=True)
    value = Column(Float)
    market_data = relationship('MarketData', back_populates='indicators')

class ModelInfo(Base):
    __tablename__ = 'model_info'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    parameters = Column(String)
    mae = Column(Float)
    rmse = Column(Float)
    predictions = relationship('Prediction', back_populates='model')

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey('model_info.id'))
    ticker_id = Column(Integer, ForeignKey('tickers.id'))
    date = Column(Date, index=True)
    predicted_close = Column(Float)
    model = relationship('ModelInfo', back_populates='predictions')
