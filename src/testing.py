from threshold import ThresholdOptimizer
import pandas as pd

threshold= ThresholdOptimizer.load()
df= pd.read_csv("models/threshold_results.csv")

print("Loaded threshold:",threshold)
print(df.sort_values("F1",ascending=False).head(10))