import pandas as pd
import numpy as np

# Simulate fake accident dataset
np.random.seed(42)

num_samples = 100  # You can increase this number as needed

df = pd.DataFrame({
    'accident_time': np.random.choice(pd.date_range('2023-01-01', periods=24, freq='H').strftime('%H:%M'), size=num_samples),
    'weather': np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], size=num_samples),
    'road_condition': np.random.choice(['Dry', 'Wet', 'Icy'], size=num_samples),
    'latitude': 51.5 + np.random.rand(num_samples) * 0.1,
    'longitude': -0.1 + np.random.rand(num_samples) * 0.1
})

print("✅ Sample accident dataset created:")
print(df.head())
