import pandas as pd
import numpy as np
import os

def generate_dubai_data(n=1000):
    np.random.seed(42)
    
    neighborhoods = {
        'Dubai Marina': 1600,
        'Downtown Dubai': 2200,
        'Jumeirah Village Circle': 900,
        'Palm Jumeirah': 3500,
        'Business Bay': 1400,
        'Dubai Hills Estate': 1800
    }
    
    data = []
    for i in range(n):
        nb = np.random.choice(list(neighborhoods.keys()))
        avg_psf = neighborhoods[nb]
        size = np.random.randint(500, 5000)
        noise = np.random.normal(0, 0.1) 
        price = int(size * avg_psf * (1 + noise))
        
        data.append({
            'id': i + 1,
            'neighborhood': nb, # THE MISSING COLUMN
            'size': size,
            'price': price,
            'description': f"Beautiful property in {nb}"
        })
        
    df = pd.DataFrame(data)
    
    # Force the path to the root 'data' folder
    os.makedirs('data', exist_ok=True)
    file_path = 'data/raw_data.csv'
    
    df.to_csv(file_path, index=False)
    
    if os.path.exists(file_path):
        print(f"✅ SUCCESS: Generated {len(df)} rows with columns: {df.columns.tolist()}")
    else:
        print("❌ CRITICAL ERROR: Could not save the file!")
        exit(1)

if __name__ == "__main__":
    generate_dubai_data()
