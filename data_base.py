import yfinance as yf
import pandas as pd
import ta


def download_market_data(ticker, start_date, end_date):
    """Descarga datos históricos de Yahoo Finance y aplana las columnas MultiIndex."""
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns] 

    return data


def calculate_indicators(df):
    """Calcula indicadores técnicos básicos usando la librería 'ta'."""
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_hist'] = ta.trend.macd_diff(df['Close'])
    df.dropna(inplace=True)  # Eliminar filas con valores NaN
    return df


if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2010-01-01"
    end_date = "2024-01-01"

    market_data = download_market_data(ticker, start_date, end_date)
    print("Columnas después de aplanar:", market_data.columns)  # Debugging

    market_data = calculate_indicators(market_data)

    print(market_data.tail())
    market_data.to_csv(f"{ticker}_market_data.csv", index=False)