import pandas as pd
import os

path = r'c:\Users\Administrator\Documents\phd\site-bateia\bateia-doc\database\Redes Neurais\TRY-1\GAP\MLPerceptron\BMMLP\Previsto\VGG16\predicoes_reais_f0_vgg16_bm_gap.csv'
df = pd.read_csv(path)
print("Columns:", list(df.columns))
print("First 5 clusters:", list(df['Cluster'].head()))
print("Spaces check:", [f"'{c}'" for c in df['Cluster'].head()])
