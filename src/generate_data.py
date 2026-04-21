import pandas as pd
import numpy as np
import os

def generate_dubai_data(n=1000):
    np.random.seed(42)
    neighborhoods = {
        'Dubai Marina': 1600, 'Downtown Dubai': 2200, 
        'JVC': 900, 'Palm Jumeirah': 3500, 
        'Business Bay': 1400, 'Dubai Hills': 1800
    }
    
    data = []
    for i in range(n):
        nb = np.random.choice(list(neighborhoods.keys()))
        size = np.random.randint(500, 5000)
        price = int(size * neighborhoods[nb] * (1 + np.random.normal(0, 0.1)))
        data.append({'id': i+1, 'neighborhood': nb, 'size': size, 'price': price})
        
    df = pd.DataFrame(data)
    
    # Force creation in the workspace root
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_data.csv', index=False)
    print(f"✅ Data Generated: {df.shape[0]} rows with columns {df.columns.tolist()}")

if __name__ == "__main__":
    generate_dubai_data()
