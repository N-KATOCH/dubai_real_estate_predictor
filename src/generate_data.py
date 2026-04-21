import pandas as pd
import numpy as np
import os

def generate_dubai_data(n=1000):
    np.random.seed(42)
    
    # Define neighborhoods
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
            'neighborhood': nb,
            'size': size,
            'price': price,
            'description': f"Beautiful property in {nb}"
        })
        
    df = pd.DataFrame(data)
    
    # 🛠️ STRONGER PATHING: Ensure the directory exists relative to this script
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, 'raw_data.csv')
    df.to_csv(file_path, index=False)
    
    print(f"✅ SUCCESSFULLY generated {n} rows at: {file_path}")
    print(f"📊 Columns created: {df.columns.tolist()}")

if __name__ == "__main__":
    generate_dubai_data()
