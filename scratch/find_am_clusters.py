import os
import re

base_path = r'c:\Users\Administrator\Documents\phd\site-bateia\bateia-doc\database\Redes Neurais\TRY-1\GAP'
clusters = set()

if os.path.exists(base_path):
    for root, dirs, files in os.walk(base_path):
        if 'AM' in root and 'Previsto' in root:
            for f in files:
                # predicoes_reais_f0_vgg16_mlp_Berkeley98_gap.csv
                if f.startswith('predicoes_reais_') and f.endswith('_gap.csv'):
                    parts = f.replace('.csv', '').split('_')
                    if len(parts) >= 7:
                        cluster = parts[-2]
                        clusters.add(cluster)

print("Found clusters:", sorted(list(clusters)))
