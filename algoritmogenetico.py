import random
import numpy as np
import pandas as pd
import yfinance as yf
from deap import creator, base, tools, algorithms
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define the stocks and download the data
stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "JPM", "JNJ", "V", "BA", "MRK", "CMCSA", "CSCO", "ABBV",
          "CVS", "INTC", "MCD", "COST", "TMUS", "BMY", "ORCL", "UNP", "MDT", "MMM", "F", "IBM", "CAT", "GE", "XOM"]
data = yf.download(stocks, start="2017-01-01", end="2017-12-31")
log_returns = np.log(data['Close'] / data['Close'].shift(1))

data_2018 = yf.download(stocks, start="2018-01-01", end="2018-12-31")
log_returns_2018 = np.log(data_2018['Close'] / data_2018['Close'].shift(1))

data_2019 = yf.download(stocks, start="2019-01-01", end="2019-12-31")
log_returns_2019 = np.log(data_2019['Close'] / data_2019['Close'].shift(1))


# Save the data to a CSV file
data.to_csv('stocks_data.csv', index=False)

# Define the evaluate functions
def evaluate(individual, log_returns):
    individual_array = np.array(individual)
    individual_array = individual_array / np.sum(individual_array)  # Normalize the weights
    portfolio_return = np.sum(individual_array * log_returns.mean() * 252)
    portfolio_std_dev = np.sqrt(np.dot(individual_array, np.dot(log_returns.cov() * 252, individual_array)))

    # Define the target return and the tolerance for exceeding the target
    target_return = 0.05
    tolerance = 0.02  # for example, we'll tolerate returns up to 6%
    risk_tolerance = 0.4  # for example, we'll tolerate a standard deviation up to 15%

    # If the portfolio return is less than the target return, apply a penalty to the portfolio return
    if portfolio_return < target_return:
        penalty = abs(portfolio_return - target_return)
        portfolio_return -= penalty
    # If the portfolio return is greater than the target return plus the tolerance, also apply a penalty
    elif portfolio_return > target_return + tolerance:
        penalty = abs(portfolio_return - (target_return + tolerance))
        portfolio_return -= penalty

    # If the portfolio standard deviation is greater than the risk tolerance, apply a penalty to the portfolio return
    if portfolio_std_dev > risk_tolerance:
        risk_penalty = abs(portfolio_std_dev - risk_tolerance)
        portfolio_return -= risk_penalty

    return portfolio_return, portfolio_std_dev


def evaluate_balanced(individual, log_returns):
    individual_array = np.array(individual)
    individual_array = individual_array / np.sum(individual_array)  # Normalize the weights
    portfolio_return = np.sum(individual_array * log_returns.mean() * 252)
    portfolio_std_dev = np.sqrt(np.dot(individual_array, np.dot(log_returns.cov() * 252, individual_array)))
    return portfolio_return, portfolio_std_dev  # Return a tuple of two values

def evaluate_risk_taker(individual, log_returns):
    individual_array = np.array(individual)
    individual_array = individual_array / np.sum(individual_array)  # Normalize the weights
    portfolio_return = np.sum(individual_array * log_returns.mean() * 252)
    portfolio_std_dev = np.sqrt(np.dot(individual_array, np.dot(log_returns.cov() * 252, individual_array)))
    return (2*portfolio_return, -portfolio_std_dev)  # Return a tuple with two values

def evaluate_risk_averse(individual, log_returns):
    individual_array = np.array(individual)
    individual_array = individual_array / np.sum(individual_array)  # Normalize the weights
    portfolio_return = np.sum(individual_array * log_returns.mean() * 252)
    portfolio_std_dev = np.sqrt(np.dot(individual_array, np.dot(log_returns.cov() * 252, individual_array)))
    return portfolio_return - 2*portfolio_std_dev, portfolio_std_dev  # Return both values


# Define the individual creation function
def create_individual(ind_size):
    return [random.uniform(0, 1) for _ in range(ind_size)]

# Create the fitness and individual classes
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)

# Create the toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", create_individual, ind_size=len(stocks))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate_risk_taker", evaluate_risk_taker)
toolbox.register("evaluate_balanced", evaluate_balanced)
toolbox.register("evaluate_risk_averse", evaluate_risk_averse)
toolbox.register("evaluate", evaluate, log_returns=log_returns)

# Register the evaluate, mate, mutate, and select functions
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# Define the constants
POP_SIZE = 200
CXPB = 0.7
MUTPB = 0.2
NGEN = 700


# Genetic algorithm
def run_evolution(evaluate_func):
    toolbox.register("evaluate", evaluate_func, log_returns=log_returns) 
    pop = toolbox.population(n=POP_SIZE)
    with tqdm(total=NGEN) as pbar:
        for g in range(NGEN):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for ind in offspring:
                for i in range(len(ind)):
                    ind[i] = max(0, min(1, ind[i]))  # Ensure allocations are between 0 and 1
                ind[:] = ind / np.sum(ind)  # Normalize the weights

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring

            pbar.update(1)
            pbar.set_description(f'Generation {g + 1}/{NGEN}')

    return tools.selBest(pop, 1)[0],pop, invalid_ind


#Run the genetic algorithm
best_ind, pop, invalid_ind = run_evolution(toolbox.evaluate)
best_ind_risk_taker, _, _ = run_evolution(toolbox.evaluate_risk_taker)
best_ind_balanced, _, _ = run_evolution(toolbox.evaluate_balanced)
best_ind_risk_averse, _, _ = run_evolution(toolbox.evaluate_risk_averse)

# Define the desired return and risk
desired_return = 0.05
desired_risk = 0.00

# Calculate the Euclidean distance between the (return, risk) of each individual and the desired (return, risk)
distances = [(ind, np.sqrt((evaluate(ind, log_returns)[0] - desired_return)**2 + (evaluate(ind, log_returns)[1] - desired_risk)**2)) for ind in pop]

# Sort the individuals by their distance to the desired (return, risk)
distances.sort(key=lambda x: x[1])

# The best individual is the one with the smallest distance
best_ind = distances[0][0]


best_individual_df = pd.DataFrame([best_ind_risk_averse], columns=stocks)
best_individual_df.to_csv("best_individual_risk_averse.csv", index=False)

best_individual_df = pd.DataFrame([best_ind_risk_taker], columns=stocks)
best_individual_df.to_csv("best_individual_risk_taker.csv", index=False)

best_individual_df = pd.DataFrame([best_ind_balanced], columns=stocks)
best_individual_df.to_csv("best_individual_balanced.csv", index=False)


# Normalize the weights
weights_sum = sum(best_ind)
weights_relative = [weight / weights_sum for weight in best_ind]

# Calculate the capital allocation
total_capital = 1000000  # Example of total available capital
capital_allocation = [weight * total_capital for weight in weights_relative]

portfolio_return_2018, portfolio_std_dev_2018 = evaluate(best_ind, log_returns_2018)
portfolio_return_2019, portfolio_std_dev_2019 = evaluate(best_ind, log_returns_2019)



# Plot the solutions
fig, ax = plt.subplots()
for ind in invalid_ind:
    ax.scatter(np.sqrt(np.dot(ind, np.dot(log_returns.cov() * 252, ind))), *evaluate(ind, log_returns), color='b')
ax.scatter(np.sqrt(np.dot(best_ind, np.dot(log_returns.cov() * 252, best_ind))), *evaluate(best_ind, log_returns), color='r')
ax.set_title('Return-Volatility Space of Solutions')
ax.set_xlabel('Volatility')
ax.set_ylabel('Return')
plt.show()

# Print the best individual and the capital allocation
portfolio_return, portfolio_std_dev = evaluate(best_ind, log_returns)
portfolio_return = round(portfolio_return, 2)
portfolio_std_dev = round(portfolio_std_dev, 2)
print("Best individual is", [round(x, 2) for x in best_ind], "(", portfolio_return, ",", portfolio_std_dev, ")")
for stock, allocation in zip(stocks, capital_allocation):
    print(stock, ":", round(allocation, 2))

# Save the best individual and the entire population to CSV files
best_individual_df = pd.DataFrame([best_ind], columns=stocks)
best_individual_df.to_csv("best_individual.csv", index=False)
population_df = pd.DataFrame(pop, columns=stocks)
population_df.to_csv("population.csv", index=False)

portfolio_weights = pd.DataFrame({
    'Stock': stocks,
    'Weight': best_ind
})
portfolio_weights.to_csv('portfolio_weights.csv', index=False)

daily_portfolio_returns = (log_returns * best_ind).sum(axis=1)
daily_portfolio_returns.to_csv('daily_portfolio_returns.csv')



# Calculate the cumulative returns of the portfolio for 2017, 2018 and 2019
daily_portfolio_returns_2017 = (log_returns * best_ind).sum(axis=1)
cumulative_returns_2017 = (1 + daily_portfolio_returns_2017).cumprod()

daily_portfolio_returns_2018 = (log_returns_2018 * best_ind).sum(axis=1)
cumulative_returns_2018 = (1 + daily_portfolio_returns_2018).cumprod()

daily_portfolio_returns_2019 = (log_returns_2019 * best_ind).sum(axis=1)
cumulative_returns_2019 = (1 + daily_portfolio_returns_2019).cumprod()

# Save the cumulative returns to CSV files
cumulative_returns_2017.to_frame('2017').to_csv('cumulative_returns_2017.csv')
cumulative_returns_2018.to_frame('2018').to_csv('cumulative_returns_2018.csv')
cumulative_returns_2019.to_frame('2019').to_csv('cumulative_returns_2019.csv')


annual_returns = log_returns.mean() * 252
annual_std_devs = log_returns.std() * np.sqrt(252)
risk_return_data = pd.DataFrame({
    'Stock': stocks,
    'AnnualReturn': annual_returns,
    'AnnualStdDev': annual_std_devs
})
risk_return_data.to_csv('risk_return_data.csv', index=False)


correlation_matrix = log_returns.corr()
correlation_matrix.to_csv('correlation_matrix.csv')

benchmark_data = yf.download('^GSPC', start="2017-01-01", end="2017-12-31")
benchmark_log_returns = np.log(benchmark_data['Close'] / benchmark_data['Close'].shift(1))
benchmark_log_returns.to_csv('benchmark_log_returns.csv')

risk_free_rate = 0.01  # Assume a risk-free rate of 1%
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

performance_data = pd.DataFrame({
    'Metric': ['TotalReturn', 'StdDev', 'SharpeRatio'],
    'Value': [portfolio_return, portfolio_std_dev, sharpe_ratio]
})
performance_data.to_csv('performance_data.csv', index=False)# Calculate the capital allocation


# Calculate the daily returns for the risk taker and balanced portfolios for each year
risk_taker_returns_2017 = (log_returns * best_ind_risk_taker).sum(axis=1)
risk_taker_returns_2018 = (log_returns_2018 * best_ind_risk_taker).sum(axis=1)
risk_taker_returns_2019 = (log_returns_2019 * best_ind_risk_taker).sum(axis=1)

balanced_returns_2017 = (log_returns * best_ind_balanced).sum(axis=1)
balanced_returns_2018 = (log_returns_2018 * best_ind_balanced).sum(axis=1)
balanced_returns_2019 = (log_returns_2019 * best_ind_balanced).sum(axis=1)

risk_averse_returns_2017 = (log_returns * best_ind_risk_averse).sum(axis=1)
risk_averse_returns_2018 = (log_returns_2018 * best_ind_risk_averse).sum(axis=1)
risk_averse_returns_2019 = (log_returns_2019 * best_ind_risk_averse).sum(axis=1)

# Calculate the cumulative returns for the risk taker, risk averse and balanced portfolios for each year
cumulative_returns_risk_taker = {
    '2017': (1 + risk_taker_returns_2017).cumprod(),
    '2018': (1 + risk_taker_returns_2018).cumprod(),
    '2019': (1 + risk_taker_returns_2019).cumprod()
}

cumulative_returns_risk_averse = {
    '2017': (1 + risk_averse_returns_2017).cumprod(),
    '2018': (1 + risk_averse_returns_2018).cumprod(),
    '2019': (1 + risk_averse_returns_2019).cumprod()
}

cumulative_returns_balanced = {
    '2017': (1 + balanced_returns_2017).cumprod(),
    '2018': (1 + balanced_returns_2018).cumprod(),
    '2019': (1 + balanced_returns_2019).cumprod()
}

# Calculate the annual returns for the risk taker, risk averse and balanced portfolios for each year
risk_taker_annual_return_2017 = risk_taker_returns_2017.mean() * 252
risk_taker_annual_return_2018 = risk_taker_returns_2018.mean() * 252
risk_taker_annual_return_2019 = risk_taker_returns_2019.mean() * 252

risk_averse_annual_return_2017 = risk_averse_returns_2017.mean() * 252
risk_averse_annual_return_2018 = risk_averse_returns_2018.mean() * 252
risk_averse_annual_return_2019 = risk_averse_returns_2019.mean() * 252

balanced_annual_return_2017 = balanced_returns_2017.mean() * 252
balanced_annual_return_2018 = balanced_returns_2018.mean() * 252
balanced_annual_return_2019 = balanced_returns_2019.mean() * 252

# Risk taker portfolio
risk_taker_total_return_2017 = cumulative_returns_risk_taker['2017'].iloc[-1] - 1
risk_taker_total_return_2018 = cumulative_returns_risk_taker['2018'].iloc[-1] - 1
risk_taker_total_return_2019 = cumulative_returns_risk_taker['2019'].iloc[-1] - 1

# Balanced portfolio
balanced_total_return_2017 = cumulative_returns_balanced['2017'].iloc[-1] - 1
balanced_total_return_2018 = cumulative_returns_balanced['2018'].iloc[-1] - 1
balanced_total_return_2019 = cumulative_returns_balanced['2019'].iloc[-1] - 1

# Calculate the Sharpe ratios for the risk taker, risk averse and balanced portfolios for each year
risk_free_rate = 0.01  # Assume a risk-free rate of 1%

risk_taker_sharpe_ratio_2017 = (risk_taker_annual_return_2017 - risk_free_rate) / risk_taker_returns_2017.std() * np.sqrt(252)
risk_taker_sharpe_ratio_2018 = (risk_taker_annual_return_2018 - risk_free_rate) / risk_taker_returns_2018.std() * np.sqrt(252)
risk_taker_sharpe_ratio_2019 = (risk_taker_annual_return_2019 - risk_free_rate) / risk_taker_returns_2019.std() * np.sqrt(252)

risk_averse_sharpe_ratio_2017 = (risk_averse_annual_return_2017 - risk_free_rate) / risk_averse_returns_2017.std() * np.sqrt(252)
risk_averse_sharpe_ratio_2018 = (risk_averse_annual_return_2018 - risk_free_rate) / risk_averse_returns_2018.std() * np.sqrt(252)
risk_averse_sharpe_ratio_2019 = (risk_averse_annual_return_2019 - risk_free_rate) / risk_averse_returns_2019.std() * np.sqrt(252)

balanced_sharpe_ratio_2017 = (balanced_annual_return_2017 - risk_free_rate) / balanced_returns_2017.std() * np.sqrt(252)
balanced_sharpe_ratio_2018 = (balanced_annual_return_2018 - risk_free_rate) / balanced_returns_2018.std() * np.sqrt(252)
balanced_sharpe_ratio_2019 = (balanced_annual_return_2019 - risk_free_rate) / balanced_returns_2019.std() * np.sqrt(252)



# Total return
total_return_df = pd.DataFrame({
    'Year': ['2017', '2018', '2019'],
    'Risk Taker': [risk_taker_total_return_2017, risk_taker_total_return_2018, risk_taker_total_return_2019],
    'Balanced': [balanced_total_return_2017, balanced_total_return_2018, balanced_total_return_2019]
})
total_return_df.to_csv('total_return.csv', index=False)

# Average annual return
annual_return_df = pd.DataFrame({
    'Year': ['2017', '2018', '2019'],
    'Risk Taker': [risk_taker_annual_return_2017, risk_taker_annual_return_2018, risk_taker_annual_return_2019],
    'Balanced': [balanced_annual_return_2017, balanced_annual_return_2018, balanced_annual_return_2019]
})
annual_return_df.to_csv('annual_return.csv', index=False)

# Sharpe ratio
sharpe_ratio_df = pd.DataFrame({
    'Year': ['2017', '2018', '2019'],
    'Risk Taker': [risk_taker_sharpe_ratio_2017, risk_taker_sharpe_ratio_2018, risk_taker_sharpe_ratio_2019],
    'Balanced': [balanced_sharpe_ratio_2017, balanced_sharpe_ratio_2018, balanced_sharpe_ratio_2019]
})
sharpe_ratio_df.to_csv('sharpe_ratio.csv', index=False)

# Cumulative returns
cumulative_returns_risk_taker_2017_df = pd.DataFrame({
    'Date': risk_taker_returns_2017.index,
    '2017': risk_taker_returns_2017.cumsum()
})

cumulative_returns_risk_taker_2018_df = pd.DataFrame({
    'Date': risk_taker_returns_2018.index,
    '2018': risk_taker_returns_2018.cumsum()
})

cumulative_returns_risk_taker_2019_df = pd.DataFrame({
    'Date': risk_taker_returns_2019.index,
    '2019': risk_taker_returns_2019.cumsum()
})

cumulative_returns_balanced_2017_df = pd.DataFrame({
    'Date': balanced_returns_2017.index,
    '2017': balanced_returns_2017.cumsum()
})

cumulative_returns_balanced_2018_df = pd.DataFrame({
    'Date': balanced_returns_2018.index,
    '2018': balanced_returns_2018.cumsum()
})

cumulative_returns_balanced_2019_df = pd.DataFrame({
    'Date': balanced_returns_2019.index,
    '2019': balanced_returns_2019.cumsum()
})

cumulative_returns_risk_averse_2017_df = pd.DataFrame({
    'Date': risk_averse_returns_2017.index,
    '2017': risk_averse_returns_2017.cumsum()
})

cumulative_returns_risk_averse_2018_df = pd.DataFrame({
    'Date': risk_averse_returns_2018.index,
    '2018': risk_averse_returns_2018.cumsum()
})

cumulative_returns_risk_averse_2019_df = pd.DataFrame({
    'Date': risk_averse_returns_2019.index,
    '2019': risk_averse_returns_2019.cumsum()
})

total_return_df['Risk Averse'] = [cumulative_returns_risk_averse['2017'].iloc[-1] - 1, cumulative_returns_risk_averse['2018'].iloc[-1] - 1, cumulative_returns_risk_averse['2019'].iloc[-1] - 1]
annual_return_df['Risk Averse'] = [risk_averse_annual_return_2017, risk_averse_annual_return_2018, risk_averse_annual_return_2019]
sharpe_ratio_df['Risk Averse'] = [risk_averse_sharpe_ratio_2017, risk_averse_sharpe_ratio_2018, risk_averse_sharpe_ratio_2019]

total_return_df.to_csv('total_return.csv', index=False)
annual_return_df.to_csv('annual_return.csv', index=False)
sharpe_ratio_df.to_csv('sharpe_ratio.csv', index=False)

# Save to CSV
cumulative_returns_risk_taker_2017_df.to_csv('cumulative_returns_risk_taker_2017.csv', index=False)
cumulative_returns_risk_taker_2018_df.to_csv('cumulative_returns_risk_taker_2018.csv', index=False)
cumulative_returns_risk_taker_2019_df.to_csv('cumulative_returns_risk_taker_2019.csv', index=False)

cumulative_returns_balanced_2017_df.to_csv('cumulative_returns_balanced_2017.csv', index=False)
cumulative_returns_balanced_2018_df.to_csv('cumulative_returns_balanced_2018.csv', index=False)
cumulative_returns_balanced_2019_df.to_csv('cumulative_returns_balanced_2019.csv', index=False)

cumulative_returns_risk_averse_2017_df.to_csv('cumulative_returns_risk_averse_2017.csv', index=False)
cumulative_returns_risk_averse_2018_df.to_csv('cumulative_returns_risk_averse_2018.csv', index=False)
cumulative_returns_risk_averse_2019_df.to_csv('cumulative_returns_risk_averse_2019.csv', index=False)


# Create a DataFrame to store the results
results = pd.DataFrame({
    'Year': ['2017', '2018', '2019'],
    'Return': [portfolio_return, portfolio_return_2018, portfolio_return_2019],
    'Risk': [portfolio_std_dev, portfolio_std_dev_2018, portfolio_std_dev_2019]
})

results.to_csv('portfolio_performance.csv', index=False)

# Risk Taker
results_risk_taker = pd.DataFrame({
    'Year': ['2017', '2018', '2019'],
    'Return': [risk_taker_annual_return_2017, risk_taker_annual_return_2018, risk_taker_annual_return_2019],
    'Risk': [risk_taker_returns_2017.std(), risk_taker_returns_2018.std(), risk_taker_returns_2019.std()]
})
results_risk_taker.to_csv('portfolio_performance_risk_taker.csv', index=False)

# Risk Averse
results_risk_averse = pd.DataFrame({
    'Year': ['2017', '2018', '2019'],
    'Return': [risk_averse_annual_return_2017, risk_averse_annual_return_2018, risk_averse_annual_return_2019],
    'Risk': [risk_averse_returns_2017.std(), risk_averse_returns_2018.std(), risk_averse_returns_2019.std()]
})
results_risk_averse.to_csv('portfolio_performance_risk_averse.csv', index=False)

# Balanced
results_balanced = pd.DataFrame({
    'Year': ['2017', '2018', '2019'],
    'Return': [balanced_annual_return_2017, balanced_annual_return_2018, balanced_annual_return_2019],
    'Risk': [balanced_returns_2017.std(), balanced_returns_2018.std(), balanced_returns_2019.std()]
})
results_balanced.to_csv('portfolio_performance_balanced.csv', index=False)

# Calculate the excess returns
excess_returns = daily_portfolio_returns - benchmark_log_returns

# Calculate the Information Ratio
information_ratio = excess_returns.mean() / excess_returns.std()

# Risk Taker
excess_returns_risk_taker_2017 = risk_taker_returns_2017 - benchmark_log_returns
information_ratio_risk_taker_2017 = excess_returns_risk_taker_2017.mean() / excess_returns_risk_taker_2017.std()

# Balanced
excess_returns_balanced_2017 = balanced_returns_2017 - benchmark_log_returns
information_ratio_balanced_2017 = excess_returns_balanced_2017.mean() / excess_returns_balanced_2017.std()

# Risk Averse
excess_returns_risk_averse_2017 = risk_averse_returns_2017 - benchmark_log_returns
information_ratio_risk_averse_2017 = excess_returns_risk_averse_2017.mean() / excess_returns_risk_averse_2017.std()

# Create a DataFrame to store the results
results = pd.DataFrame({
    'Portfolio': ['Overall', 'Risk Taker', 'Balanced', 'Risk Averse'],
    'Information Ratio': [information_ratio, information_ratio_risk_taker_2017, information_ratio_balanced_2017, information_ratio_risk_averse_2017]
})

# Save the results to a CSV file
results.to_csv('information_ratio.csv', index=False)