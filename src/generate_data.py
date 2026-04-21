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
        # Price = (Size * Base PSF) + 10% random noise
        price = int(size * neighborhoods[nb] * (1 + np.random.normal(0, 0.1)))
        data.append({'id': i+1, 'neighborhood': nb, 'size': size, 'price': price})
        
    df = pd.DataFrame(data)
    
    # 🛠️ THE FIX: Create the 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print("📁 Created 'data' directory")
    
    file_path = 'data/raw_data.csv'
    df.to_csv(file_path, index=False)
    
    print(f"✅ Data Generated: {df.shape[0]} rows saved to {file_path}")

if __name__ == "__main__":
    generate_dubai_data()
