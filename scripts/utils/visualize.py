import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# --- Akademik Stil Ayarları ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

def get_core_model(name):
    """Model adından suffixleri temizleyerek ana kökü bulur."""
    if pd.isna(name) or name == 'NA': return 'NA'
    name = str(name)
    # it, instruct, reasoning, think vb. takıları ve sonrasını temizle
    clean = re.sub(r'[-_](base|it|instruct|reasoning|hf|think).*$', '', name, flags=re.IGNORECASE)
    return clean.strip()

def get_variant_label(name):
    """Modelin hangi varyant (Base, Instruct, Think) olduğunu belirler."""
    name = str(name).lower()
    if 'think-true' in name: return 'Think-True'
    if 'think-false' in name: return 'Think-False'
    # Gemma/Llama gibi modellerde 'it' veya 'instruct' varsa ayır
    if '-it' in name or 'instruct' in name: return 'Instruct'
    return 'Base'

def process_and_plot(input_csv="results/summary_results.csv", output_base_dir="results/plots/"):
    # Klasör hazırlığı
    dir_summary = os.path.join(output_base_dir, "summary_matrices")
    os.makedirs(dir_summary, exist_ok=True)

    df = pd.read_csv(input_csv)
    
    # Eksik verileri doldurma
    for col in ['judge_model', 'type', 'dataset', 'base_model']:
        df[col] = df[col].fillna('NA')
    
    # Kök model ve Varyant belirleme
    df['core_model'] = df['base_model'].apply(get_core_model)
    df['variant'] = df['base_model'].apply(get_variant_label)

    # Melt işlemi (Pass@k sütunlarını tek sütuna indirger)
    pass_cols = [col for col in df.columns if col.startswith('pass@')]
    df_melted = df.melt(
        id_vars=['type', 'dataset', 'base_model', 'core_model', 'judge_model', 'variant'],
        value_vars=pass_cols, var_name='metric_raw', value_name='score'
    )
    df_melted['k'] = df_melted['metric_raw'].str.extract(r'(\d+)').astype(int)

    # =============================================================
    # 1. DUPLICATE TEMİZLİĞİ (Overlap Engelleme)
    # =============================================================
    keys = ['dataset', 'core_model', 'judge_model', 'variant', 'k']
    
    # Aynı kategoride birden fazla kayıt varsa ortalamasını alarak çizgiyi teke düşürür
    df_clean = df_melted.groupby(keys, as_index=False).agg({'score': 'mean'})
    
    # Özet matrisler için Ministral'i çıkar
    s_df = df_clean[~df_clean['core_model'].str.contains('Ministral', case=False, na=False)].copy()

    # Stil tanımları
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    default_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    
    def create_style_map(items):
        return {item: {'color': 'black' if item == 'Base' else default_colors[i % len(default_colors)],
                       'marker': 'o' if item == 'Base' else default_markers[i % len(default_markers)]} 
                for i, item in enumerate(items)}

    global_styles = {
        'variant': create_style_map(['Base', 'Think-False', 'Think-True', 'Instruct']),
        'dataset': create_style_map(s_df['dataset'].unique()),
        'judge_model': create_style_map(s_df['judge_model'].unique())
    }

    # =============================================================
    # 2. GÜNCELLENMİŞ ÖZET MATRİS FONKSİYONU
    # =============================================================
    def plot_matrix_grid(data, split_by, row_by, col_by, line_by, prefix, title_main):
        # Base modelleri referans olarak kullanmak için ayır
        base_pool = data[data['variant'] == 'Base'].copy()

        for split_val in data[split_by].unique():
            # DİKKAT: Eski koddaki 'NA' değerlerini atlayan satır kaldırıldı.
            # Artık "judge_model = NA" (Hiç judge olmayan hal) için de grafik üretilecek.
            
            subset_split = data[data[split_by] == split_val].copy()
            if subset_split.empty: continue
            
            rows = sorted(subset_split[row_by].unique())
            cols = sorted(subset_split[col_by].unique())
            
            fig, axes = plt.subplots(len(rows), len(cols), figsize=(5*len(cols), 4.5*len(rows)), 
                                     squeeze=False, sharex=True)
            
            legend_tracker = {} 
            
            for r, row_val in enumerate(rows):
                for c, col_val in enumerate(cols):
                    ax = axes[r, c]
                    
                    # Hücre verisi (Örn: Qwen-32B için Think-True/False verileri)
                    cell_data = subset_split[(subset_split[row_by] == row_val) & 
                                             (subset_split[col_by] == col_val)].copy()
                    
                    # EĞER Varyant karşılaştırması yapıyorsak, Base modeli her hücreye enjekte et
                    if prefix == "VARIANT_EFFECT":
                        cell_base = base_pool[(base_pool[row_by] == row_val) & 
                                              (base_pool[col_by] == col_val)]
                        
                        # Base modeli çizdirirken önce şu anki judge filtremizi deniyoruz
                        cell_base_filtered = cell_base[cell_base[split_by] == split_val]
                        
                        # Eğer bu judge için Base modeli yoksa, Base'in orijinal 'NA' (no judge) halini referans alıyoruz
                        if cell_base_filtered.empty and not cell_base.empty:
                            cell_base_filtered = cell_base[cell_base['judge_model'] == 'NA']
                            
                        cell_data = pd.concat([cell_data, cell_base_filtered]).drop_duplicates(subset=['k', line_by])

                    if not cell_data.empty:
                        for l_val in sorted(cell_data[line_by].unique()):
                            l_plot = cell_data[cell_data[line_by] == l_val].sort_values('k')
                            style = global_styles.get(line_by, {}).get(l_val, {'color': 'gray', 'marker': '.'})
                            
                            line, = ax.plot(l_plot['k'], l_plot['score'], 
                                            label=l_val, color=style['color'], 
                                            marker=style['marker'], linewidth=1.5, 
                                            markersize=5, alpha=0.8)
                            
                            if l_val not in legend_tracker:
                                legend_tracker[l_val] = line
                        
                        ax.set_title(f"{row_val} | {col_val}", fontsize=10)
                        ax.set_xscale('log', base=2)
                        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
                        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                        ax.grid(True, alpha=0.2, linestyle='--')
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', alpha=0.3)
                    
                    if r == len(rows)-1: ax.set_xlabel('k')
                    if c == 0: ax.set_ylabel('Pass@k')

            # Legend Temizliği (Her etiketten sadece 1 tane)
            if legend_tracker:
                sorted_keys = sorted(legend_tracker.keys())
                fig.legend([legend_tracker[k] for k in sorted_keys], sorted_keys, 
                           loc='lower center', ncol=len(sorted_keys), 
                           bbox_to_anchor=(0.5, -0.02), frameon=False)
            
            # Başlıklarda "NA" yerine daha şık durması için "No Judge" yazdırma
            display_split = "No Judge" if split_val == 'NA' else split_val
            plt.suptitle(f"{title_main}\n(Filter: {split_by} = {display_split})", fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Dosya adında yine NA kalabilir
            plt.savefig(os.path.join(dir_summary, f"{prefix}_{split_val}.png".replace("/", "_")), dpi=200, bbox_inches='tight')
            plt.close(fig)

    # Matrisleri Oluştur
    print("🚀 Matrisler oluşturuluyor...")
    plot_matrix_grid(s_df, 'judge_model', 'dataset', 'core_model', 'variant', "VARIANT_EFFECT", "Variant Comparison (Base vs Think vs Instruct)")
    plot_matrix_grid(s_df, 'judge_model', 'variant', 'core_model', 'dataset', "LANGUAGE_EFFECT", "Language Impact Analysis")
    plot_matrix_grid(s_df, 'variant', 'dataset', 'core_model', 'judge_model', "JUDGE_EFFECT", "Judge Model Comparison")
    
    print(f"✅ İşlem tamamlandı. Çıktılar: {dir_summary}")

if __name__ == "__main__":
    process_and_plot()