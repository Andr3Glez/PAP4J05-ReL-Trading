import pandas as pd
import numpy as np
from deepQLearning import TradingGymEnv, train_deep_q_learning, train_ppo, evaluate_model
import matplotlib.pyplot as plt
from data_base import download_and_prepare_data_by_years
import os
from stable_baselines3 import PPO

def load_market_data_by_years(ticker="AAPL", start_date="2010-01-01", end_date="2024-01-01"):
    # Descargar y preparar los datos del mercado, divididos por año
    yearly_data, full_data = download_and_prepare_data_by_years(ticker, start_date, end_date)
    return yearly_data, full_data

def plot_performance(model, env, full_data):
    # Crear un ambiente de prueba separado usando el dataset completo
    test_env = TradingGymEnv({0: full_data}, initial_balance=env.initial_balance, trading_fee=env.trading_fee)

    # Simular episodio con el modelo entrenado
    obs, _ = test_env.reset()
    done = False
    truncated = False
    total_reward = 0
    portfolio_values = [test_env.initial_balance]
    actions_taken = []
    prices = []

    while not (done or truncated):
        action, _ = model.predict(obs)
        actions_taken.append(action)
        price = test_env.data.iloc[test_env.current_step]['Close']
        prices.append(price)

        obs, reward, done, truncated, _ = test_env.step(action)
        total_reward += reward

        portfolio_value = test_env.balance + test_env.shares_held * price
        portfolio_values.append(portfolio_value)

    # === Gráfico 1: Valor del portafolio ===
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Valor del Portafolio')
    plt.title('Valor del Portafolio Durante Simulación')
    plt.xlabel('Paso')
    plt.ylabel('Valor del Portafolio ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    plt.close()

    # === Gráfico 2: Precio + Buy/Sell ===
    buy_signals = [i for i, a in enumerate(actions_taken) if a == 1]
    sell_signals = [i for i, a in enumerate(actions_taken) if a == 2]

    plt.figure(figsize=(14, 6))
    plt.plot(prices, label='Precio', linewidth=1)
    plt.scatter(buy_signals, [prices[i] for i in buy_signals], marker='^', color='green', label='Buy', s=60)
    plt.scatter(sell_signals, [prices[i] for i in sell_signals], marker='v', color='red', label='Sell', s=60)
    plt.title('Acciones del Agente sobre Precio de AAPL')
    plt.xlabel('Paso')
    plt.ylabel('Precio ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('acciones_agente.png')
    plt.close() 

    # Calcular métricas de performance
    metrics = test_env.calculate_performance_metrics()
    return total_reward, portfolio_values, metrics

def compute_buy_and_hold(full_data, initial_balance=10000):
    first_price = full_data.iloc[0]['Close']
    last_price = full_data.iloc[-1]['Close']
    
    shares_bought = initial_balance / first_price
    final_value = shares_bought * last_price
    
    total_return = (final_value / initial_balance) - 1
    annualized_return = (1 + total_return) ** (1 / ((len(full_data) / 252))) - 1

    return {
        'BuyHold_Final_Value': final_value,
        'BuyHold_Total_Profit': final_value - initial_balance,
        'BuyHold_Annualized_Return': annualized_return
    }


def main():
    # Cargar datos de mercado
    yearly_data, full_data = load_market_data_by_years()

    # Dividir datos en train (2010–2020) y validación (2021–2023)
    train_data = {k: v for k, v in yearly_data.items() if k <= 2020}
    valid_data = {k: v for k, v in yearly_data.items() if 2021 <= k <= 2023}

    env = TradingGymEnv(train_data)
    valid_env = TradingGymEnv(valid_data, initial_balance=env.initial_balance, trading_fee=env.trading_fee)

    # Configurar el directorio de logs para TensorBoard
    log_dir = "logs/ppo_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    # Entrenar modelo PPO con soporte de TensorBoard
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=500_000, log_interval=10)

    # Guardar el modelo entrenado
    model.save("trading_ppo_model")

    # Evaluar modelo
    eval_metrics = evaluate_model(model, env, num_episodes=50)

    # Resultados de la evaluación
    print("\nModel Evaluation Results:")
    print(f"Mean Reward: {eval_metrics['mean_reward']:.4f} ± {eval_metrics['std_reward']:.4f}")
    print(f"Mean Return: {eval_metrics['mean_return']*100:.2f}% ± {eval_metrics['std_return']*100:.2f}%")
    print(f"Action Distribution: Hold={eval_metrics['action_distribution']['Hold']*100:.1f}%, " + 
        f"Buy={eval_metrics['action_distribution']['Buy']*100:.1f}%, " +
      f"Sell={eval_metrics['action_distribution']['Sell']*100:.1f}%")

    # Simular rendimiento
    total_reward, portfolio_values, metrics = plot_performance(model, env, full_data)

    # Metricas de rendimiento
    print("\nFull Dataset Performance:")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['Sortino_Ratio']:.4f}")
    print(f"Calmar Ratio: {metrics['Calmar_Ratio']:.4f}")
    print(f"Maximum Drawdown: {metrics['Max_Drawdown']*100:.2f}%")
    print(f"Win/Loss Ratio: {metrics['Win_Loss_Ratio']:.2f}")
    print(f"Annualized Return: {metrics['Annualized_Return']*100:.2f}%")
    print(f"Total Profit: ${metrics['Total_Profit']:.2f}")
    print(f"Final Portfolio Value: ${metrics['Final_Portfolio_Value']:.2f}")

    # Comparar contra Buy & Hold
    buy_hold_results = compute_buy_and_hold(full_data)
    
    print("\nBuy & Hold Benchmark:")
    print(f"Final Value: ${buy_hold_results['BuyHold_Final_Value']:.2f}")
    print(f"Total Profit: ${buy_hold_results['BuyHold_Total_Profit']:.2f}")
    print(f"Annualized Return: {buy_hold_results['BuyHold_Annualized_Return']*100:.2f}%")

    # Guardar métricas en archivo CSV
    metrics_df = pd.DataFrame({
        'Mean_Reward': [eval_metrics['mean_reward']],
        'Reward_Std': [eval_metrics['std_reward']],
        'Mean_Return': [eval_metrics['mean_return']],
        'Return_Std': [eval_metrics['std_return']],
        'Total_Reward': [total_reward],
        'Sharpe_Ratio': [metrics['Sharpe_Ratio']],
        'Sortino_Ratio': [metrics['Sortino_Ratio']],
        'Calmar_Ratio': [metrics['Calmar_Ratio']],
        'Max_Drawdown': [metrics['Max_Drawdown']],
        'Win_Loss_Ratio': [metrics['Win_Loss_Ratio']],
        'Annualized_Return': [metrics['Annualized_Return']],
        'Total_Profit': [metrics['Total_Profit']],
        'Final_Portfolio_Value': [metrics['Final_Portfolio_Value']],
        'Hold_Pct': [eval_metrics['action_distribution']['Hold']],
        'Buy_Pct': [eval_metrics['action_distribution']['Buy']],
        'Sell_Pct': [eval_metrics['action_distribution']['Sell']]
    })
    metrics_df.to_csv('performance_metrics.csv', index=False)

if __name__ == "__main__":
    main()
