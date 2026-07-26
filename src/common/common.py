import pandas as pd

SEED = 42

def load_data():
	df = pd.read_csv('data/gym_churn_us.csv')
	return df