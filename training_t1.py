import pandas as pd
import numpy as np
from environment import TradingEnvironment
from deepQLearning import TradingGymEnv, train_deep_q_learning, evaluate_model
import matplotlib.pyplot as plt

def load_market_data(ticker="AAPL", start_date="2010-01-01", end_date="2024-01-01"):
    from data_base import download_market_data, calculate_indicators
    
    # Descargar y preparar datos de mercado
    market_data = download_market_data(ticker, start_date, end_date)
    market_data = calculate_indicators(market_data)
    
    return market_data

def plot_performance(model, env):
    # Simular episodio con el modelo entrenado
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    portfolio_values = [env.initial_balance]

    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        portfolio_values.append(env.balance + (env.shares_held * env.data.loc[env.current_step, 'Close']))

    # Graficar valor del portafolio
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values)
    plt.title('Valor del Portafolio Durante Simulación')
    plt.xlabel('Pasos')
    plt.ylabel('Valor del Portafolio')
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    plt.close()

    return total_reward, portfolio_values

def main():
    # Cargar datos de mercado
    market_data = load_market_data()

    # Crear entorno de trading
    env = TradingGymEnv(market_data)

    # Entrenar modelo Deep Q-Learning
    model = train_deep_q_learning(env)

    # Guardar modelo
    model.save("trading_dqn_model")

    # Evaluar modelo
    mean_reward, std_reward = evaluate_model(model, env)

    # Simular rendimiento
    total_reward, portfolio_values = plot_performance(model, env)

    # Imprimir resultados
    print("\nResumen de Rendimiento:")
    print(f"Recompensa Media: {mean_reward}")
    print(f"Desviación Estándar de Recompensa: {std_reward}")
    print(f"Recompensa Total: {total_reward}")

    # Guardar métricas en archivo CSV
    metrics_df = pd.DataFrame({
        'Mean_Reward': [mean_reward],
        'Reward_Std': [std_reward],
        'Total_Reward': [total_reward]
    })
    metrics_df.to_csv('performance_metrics.csv', index=False)

if __name__ == "__main__":
    main()