import numpy as np
import pandas as pd
import os

def generate_synthetic_cmapss(filename, num_units=10, max_cycles=200):
    np.random.seed(42)
    data = []
    
    for unit_id in range(1, num_units + 1):
        # Each unit has a random lifespan between 150 and max_cycles
        lifespan = np.random.randint(150, max_cycles + 1)
        
        for cycle in range(1, lifespan + 1):
            # Baseline sensors
            # Some sensors are constant, some have noise, some have trends
            s1 = 518.67  # constant
            # s2 has a linear trend
            s2 = 641.81 - 5 * (cycle / lifespan) + np.random.normal(0, 0.1)

            s3 = 1589.70 + np.random.normal(0, 0.5) # noise
            
            # s4 has a growing trend near failure (exponential)
            trend = (cycle / lifespan) ** 2
            # Increased trend magnitude from 10 to 40 to make it more distinct
            s4 = 1400.60 + 40 * trend + np.random.normal(0, 0.2)
            
            # s5 is constant
            s5 = 14.62
            
            # s6 has a slight trend
            # Increased trend magnitude from 0.1 to 2
            s6 = 21.61 + 2 * trend + np.random.normal(0, 0.01)
            
            # settings
            set1 = np.random.normal(0, 0.001)
            set2 = np.random.normal(0, 0.001)
            set3 = 100.0
            
            row = [unit_id, cycle, set1, set2, set3, s1, s2, s3, s4, s5, s6]
            # Add placeholders for other sensors to reach 21 if needed, 
            # but 6 sensors is enough for a demo
            data.append(row)
            
    columns = ['unit_id', 'cycle', 'setting1', 'setting2', 'setting3', 
               's1', 's2', 's3', 's4', 's5', 's6']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, sep=' ', index=False, header=False)
    print(f"Generated {filename}")

if __name__ == "__main__":
    os.makedirs('data/cmapss', exist_ok=True)
    generate_synthetic_cmapss('data/cmapss/train_FD001.txt', num_units=100)
    generate_synthetic_cmapss('data/cmapss/test_FD001.txt', num_units=20)
