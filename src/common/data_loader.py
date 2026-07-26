import pandas as pd

def load_data():
	df = pd.read_csv('data/gym_churn_us.csv')
	return df