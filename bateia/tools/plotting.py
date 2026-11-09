# bateia/tools/plotting.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from matplotlib.lines import Line2D

class BateiaPlotter:
    """
    Classe base para padronização de gráficos (DRY).
    Centraliza a configuração de estilo, eixos e salvamento.
    """
    def __init__(self, plot_config: Dict[str, Any]):
        self.config = plot_config
        plt.style.use(self.config.get("style", "seaborn-v0_8-whitegrid"))
        
    def _create_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        return plt.subplots(figsize=self.config.get("figsize", (8, 6)))
        
    def _save_and_close(self, fig: plt.Figure, output_path: str, filename: str):
        os.makedirs(output_path, exist_ok=True)
        fig.tight_layout()
        plt.savefig(os.path.join(output_path, filename), dpi=self.config.get("dpi", 300))
        plt.close(fig)

    def generate_scatter_comparison(self, param: str, df_dados: pd.DataFrame, output_path: str, lit_label: str = "Angelo et al.", bateia_label: str = "BATEIA-1", numbered: bool = False):
        ylabel_map = {'Rc': 'Core radius ($r_c$)', 'Rt': 'Tidal radius ($r_t$)', 'f0': r'Central density ($\sigma_0$)', 'fbg': r'Background density ($\sigma_{bg}$)'}
        color_map = {'f0': 'blue', 'Rc': 'green', 'Rt': 'red', 'fbg': 'gold'}
        
        col_x, col_y, col_err_y = f'{param}_bateia', f'{param}_lit', f'err_{param}_lit'
        if col_x not in df_dados.columns or col_y not in df_dados.columns: return

        valid_mask = ~(pd.isna(df_dados[col_x]) | pd.isna(df_dados[col_y]))
        df_valid = df_dados[valid_mask].copy()
        if df_valid.empty: return

        x_val, y_val = df_valid[col_x].values, df_valid[col_y].values
        y_err = df_valid.get(col_err_y, pd.Series(np.zeros(len(df_valid)), index=df_valid.index)).values
        x_err = np.where((y_val > 0), (y_err / y_val) * x_val, 0)

        fig, ax = self._create_figure()
        point_color = color_map.get(param, 'black')
        
        ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, fmt='none', ecolor='gray', capsize=self.config.get("error_bar_capsize", 3), alpha=0.6, zorder=1)
        
        if numbered:
            aglomerados = sorted(df_valid['Cluster'].unique())
            mapa = {c: i+1 for i, c in enumerate(aglomerados)}
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, 'mapeamento.txt'), 'w') as f:
                for c, n in mapa.items(): f.write(f"[{n:02d}] -> {c}\n")
            
            for _, row in df_valid.iterrows():
                cx, cy, num = row[col_x], row[col_y], mapa[row['Cluster']]
                ax.plot(cx, cy, marker='o', color=point_color, markersize=14, ls='None', zorder=2)
                ax.text(cx, cy, str(num), color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)
        else:
            ax.plot(x_val, y_val, marker=self.config.get("cluster_marker", 'o'), color=point_color, ls='None', zorder=2, markersize=self.config.get("marker_size", 6))

        min_val = min((x_val - x_err).min(), (y_val - y_err).min())
        max_val = max((x_val + x_err).max(), (y_val + y_err).max())
        margin = max_val * 0.1
        lim_min, lim_max = min_val - margin, max_val + margin
        
        ax.axline((lim_min, lim_min), slope=1, color='black', ls='--', alpha=0.5, zorder=1)
        ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max); ax.set_aspect('equal', adjustable='box')
        ax.set_title(ylabel_map.get(param, param), fontsize=self.config.get("label_fontsize", 12)+2, pad=15)
        ax.set_xlabel(bateia_label, fontsize=self.config.get("label_fontsize", 12)); ax.set_ylabel(lit_label, fontsize=self.config.get("label_fontsize", 12))
        
        suf = "_NUMERADO" if numbered else ""
        self._save_and_close(fig, output_path, f'scatter_comparacao_1to1{suf}_{param}.jpg')

def funcao_king(r, f0, Rc, Rt, fbg):
    if Rc <= 0 or Rt <= Rc: return np.full_like(r, fbg if fbg > 0 else 0)
    try:
        termo_r, termo_t = np.sqrt(1 + (r / Rc)**2), np.sqrt(1 + (Rt / Rc)**2)
        mask = (r <= Rt)
        king_term = np.zeros_like(r, dtype=float)
        king_term[mask] = (1.0 / termo_r[mask]) - (1.0 / termo_t)
        king_term[king_term < 0] = 0
        return fbg + f0 * (king_term**2)
    except: return np.full_like(r, fbg if fbg > 0 else 0)
