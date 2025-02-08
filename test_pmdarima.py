import pytest
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from pmdarima import model_selection


# 1. Testando funcionalidades básicas e avançadas
def test_basic_model_fit():
    data = np.random.rand(100)
    model = auto_arima(data, seasonal=False)
    assert model.aic() is not None

def test_model_update():
    data = np.random.rand(100)
    model = auto_arima(data, seasonal=False)
    new_data = np.random.rand(10)
    model.update(new_data)
    assert model.nobs_ == 110  # Verifica se o número de observações foi atualizado

def test_model_predict():
    data = np.random.rand(100)
    model = auto_arima(data, seasonal=False)
    forecast = model.predict(n_periods=10)
    assert len(forecast) == 10


# 2. Testando com casos extremos
def test_small_dataset():
    data = np.random.rand(5)
    model = auto_arima(data, seasonal=False)
    assert model is not None  # Confirma que o modelo foi criado sem erros

def test_large_dataset():
    data = np.random.rand(1000)
    model = auto_arima(data, seasonal=False)
    assert model.aic() is not None

def test_complex_seasonality():
    data = np.sin(np.linspace(0, 50, 200)) + np.sin(np.linspace(0, 100, 200))
    model = auto_arima(data, seasonal=True, m=12)
    assert model.aic() is not None


# 3. Testando integração com outras bibliotecas
def test_with_pandas_series():
    data = pd.Series(np.random.rand(100))
    model = auto_arima(data, seasonal=False)
    assert model.aic() is not None

def test_with_numpy_array():
    data = np.random.rand(100)
    model = auto_arima(data, seasonal=False)
    assert model.aic() is not None

def test_integration_with_sklearn():
    from sklearn.model_selection import train_test_split
    data = np.random.rand(100)
    train, test = train_test_split(data, test_size=0.2)
    model = auto_arima(train, seasonal=False)
    forecast = model.predict(n_periods=len(test))
    assert len(forecast) == len(test)
