import pandas as pd
import numpy as np
import os

def generate_dubai_data(n=1000):
    np.random.seed(42)
    
    # Neighborhoods and their approx price per sqft (AED)
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
        
        # Random size between 500 and 5000 sqft
        size = np.random.randint(500, 5000)
        
        # Price = (Size * Base PSF) + some random market noise
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
    
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_data.csv', index=False)
    print(f"✅ Generated {n} realistic Dubai property records with 'neighborhood' column.")

if __name__ == "__main__":
    generate_dubai_data()
