"""Notebook section: data setup."""

# One-year rating migration probabilities and end-of-year values for BBB names.
ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'Default']
probabilities = np.array([0.02, 0.33, 5.95, 86.93, 5.30, 1.17, 0.12, 0.18]) / 100
values = np.array([109.37, 109.19, 108.66, 107.55, 102.02, 98.10, 83.64, 51.13])

BBB_RATING_INDEX = 3
initial_bond_value = values[BBB_RATING_INDEX]

rating_data = pd.DataFrame({
    'Rating': ratings,
    'Probability (%)': probabilities * 100,
    'Value ($)': values,
})

print('Rating Transition Data for BBB Bonds:')
print(rating_data.to_string(index=False))
print(f'\nInitial BBB Bond Value: ${initial_bond_value:.2f}')
print(f'Sum of probabilities: {probabilities.sum():.4f}')

# Cumulative probabilities drive inverse-transform sampling in the copula method.
cumulative_probs = np.cumsum(probabilities)
print('\nCumulative probabilities:')
for rating, cum_prob in zip(ratings, cumulative_probs):
    print(f'{rating:8s}: {cum_prob:.6f}')
