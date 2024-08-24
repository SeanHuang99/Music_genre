import pandas as pd

# Load the provided pickle file
df = pd.read_pickle('../data/songs_dataframe.pkl')
pd.set_option('display.max_columns', None)
# Display the first few rows of the dataframe to understand its structure
print(df.head(30))
