import numpy as np

def create_sliding_windows(df, features, window_size=30):

    X = []

    for engine_id, engine_data in df.groupby("engine_id"):

        engine_data = engine_data.sort_values("cycle")

        values = engine_data[features].values

        for i in range(len(values) - window_size):
            window = values[i:i+window_size]
            X.append(window)

    return np.array(X)
