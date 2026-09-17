"""
한국 Fama-French 3-Factor Model v2
학술지 품질 시각화 생성 스크립트

8개 시각화:
1. 누적수익률 차트 (Cumulative Returns)
2. 팩터 프리미엄 막대그래프 (Factor Premium Bar Chart)
3. 25 포트폴리오 히트맵 (25 Portfolio Heatmap)
4. 팩터 상관관계 행렬 (Factor Correlation Matrix)
5. 60개월 이동 평균 팩터 프리미엄 (Rolling 60-month Factor Premiums)
6. 알파 분포 (Alpha Distribution)
7. 베타 분포 (Beta Distribution)
8. 미국 vs 한국 비교 (US vs Korea Comparison)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats

# ------------------------------------------------------------------
# 학술 스타일 설정
# ------------------------------------------------------------------
# 한국어 지원을 위한 Noto Sans KR 직접 로드
font_path = r'C:\Windows\Fonts\NotoSansKR-VF.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

# 학술 색상 팔레트
ACADEMIC_COLORS = {
    'Mkt-RF': '#2E5F8A',      # 딥 블루
    'SMB': '#7A8B99',         # 그레이 블루
    'HML': '#B85C38',         # 뮤티드 오렌지/브라운
    'US': '#4A6FA5',          # 미국 블루
    'KR': '#8B4513',          # 한국 브라운
    'positive': '#2E5F8A',
    'negative': '#B85C38',
    'neutral': '#7A8B99',
    'grid': '#E0E0E0',
    'bg': '#FAFAFA'
}

# 데이터 경로
DATA_DIR = r'C:\Users\PC\Desktop\AI\famakor\korean_ff3_v2\data'
OUTPUT_DIR = r'C:\Users\PC\Desktop\AI\famakor\korean_ff3_v2\output'
CHARTS_DIR = r'C:\Users\PC\AppData\Local\Temp\korean-ff3-v2-repo\output\charts'

os.makedirs(CHARTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------------
factors = pd.read_csv(os.path.join(DATA_DIR, 'factors_korea.csv'))
factors['date'] = pd.to_datetime(factors['date'])

regression = pd.read_csv(os.path.join(OUTPUT_DIR, 'regression_korea.csv'))

us_vs_kr = pd.read_csv(os.path.join(OUTPUT_DIR, 'us_vs_korea_factor_summary.csv'), comment='#')

corr_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'factor_correlations.csv'), comment='#')

portfolios_25 = pd.read_csv(os.path.join(DATA_DIR, 'portfolios_25_korea.csv'))
portfolios_25['Date'] = pd.to_datetime(portfolios_25['Date'])

# ------------------------------------------------------------------
# 헬퍼 함수
# ------------------------------------------------------------------
def save_fig(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {path}")

# ------------------------------------------------------------------
# 1. 누적수익률 차트 (Cumulative Returns)
# ------------------------------------------------------------------
def chart_cumulative_returns():
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=ACADEMIC_COLORS['bg'])
    ax.set_facecolor(ACADEMIC_COLORS['bg'])

    for col, color in [('Mkt-RF', ACADEMIC_COLORS['Mkt-RF']),
                        ('SMB', ACADEMIC_COLORS['SMB']),
                        ('HML', ACADEMIC_COLORS['HML'])]:
        cum = (1 + factors[col]).cumprod()
        ax.plot(factors['date'], cum, label=col, color=color, linewidth=1.8)

    ax.set_title('한국 Fama-French 3-Factor 누적수익률 (2000.07 ~ 2026.05)',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('연도', fontsize=12)
    ax.set_ylabel('누적수익률 (원금=1)', fontsize=12)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#333333',
              fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)

    # 주요 이벤트 주석
    events = {
        '2008-09-01': '리먼 브라더스',
        '2020-03-01': 'COVID-19',
        '2022-06-01': '금리 인상'
    }
    for date_str, label in events.items():
        date = pd.to_datetime(date_str)
        if date >= factors['date'].min() and date <= factors['date'].max():
            ax.axvline(date, color='#999999', linestyle=':', alpha=0.7, linewidth=1)
            ax.text(date, ax.get_ylim()[1]*0.95, label, rotation=90, va='top',
                    fontsize=8, color='#666666')

    save_fig(fig, '01_cumulative_returns.png')

# ------------------------------------------------------------------
# 2. 팩터 프리미엄 막대그래프 (Factor Premium Bar Chart with 95% CI)
# ------------------------------------------------------------------
def chart_factor_premium():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=ACADEMIC_COLORS['bg'])
    ax.set_facecolor(ACADEMIC_COLORS['bg'])

    factor_names = ['Mkt-RF', 'SMB', 'HML']
    means = [factors[f].mean() * 100 for f in factor_names]
    stds = [factors[f].std() * 100 for f in factor_names]
    n = len(factors)
    ci_95 = [1.96 * s / np.sqrt(n) for s in stds]

    colors = [ACADEMIC_COLORS['Mkt-RF'], ACADEMIC_COLORS['SMB'], ACADEMIC_COLORS['HML']]
    bars = ax.bar(factor_names, means, color=colors, edgecolor='#333333', linewidth=0.8, width=0.6)
    ax.errorbar(factor_names, means, yerr=ci_95, fmt='none', color='#333333', capsize=6, capthick=1.5)

    # 값 레이블
    for bar, mean, ci in zip(bars, means, ci_95):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + ci + 0.1,
                f'{mean:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(0, color='#333333', linewidth=0.8)
    ax.set_title('월간 팩터 프리미엄 (평균 ± 95% 신뢰구간)', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('월간 수익률 (%)', fontsize=12)
    ax.set_xlabel('팩터', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)

    save_fig(fig, '02_factor_premium_bar.png')

# ------------------------------------------------------------------
# 3. 25 포트폴리오 히트맵 (Average Excess Returns)
# ------------------------------------------------------------------
def chart_25_heatmap():
    # RF 병합
    pf = portfolios_25.copy()
    rf_map = dict(zip(factors['date'].dt.strftime('%Y-%m-%d'), factors['rf']))
    pf['rf'] = pf['Date'].dt.strftime('%Y-%m-%d').map(rf_map)

    # 초과수익률 계산
    cols = [c for c in pf.columns if c not in ['Date', 'rf']]
    for c in cols:
        pf[c] = (pf[c] - pf['rf']) * 100

    avg_excess = pf[cols].mean()

    # 5x5 행렬 구성
    size_labels = ['Small', '2', '3', '4', 'Big']
    bm_labels = ['Low', '2', '3', '4', 'High']
    matrix = np.zeros((5, 5))
    for i, s in enumerate(['S1', 'S2', 'S3', 'S4', 'S5']):
        for j, b in enumerate(['B1', 'B2', 'B3', 'B4', 'B5']):
            matrix[i, j] = avg_excess.get(f'{s}{b}', np.nan)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=ACADEMIC_COLORS['bg'])
    ax.set_facecolor(ACADEMIC_COLORS['bg'])

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, center=0,
                xticklabels=bm_labels, yticklabels=size_labels,
                linewidths=0.5, linecolor='white', cbar_kws={'label': '월간 초과수익률 (%)'},
                ax=ax, vmin=-2, vmax=2)

    ax.set_title('25 Size × BM 포트폴리오 평균 초과수익률 (%)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Book-to-Market (BM) 분위', fontsize=12)
    ax.set_ylabel('시가총액 (Size) 분위', fontsize=12)

    save_fig(fig, '03_heatmap_25_portfolios.png')

# ------------------------------------------------------------------
# 4. 팩터 상관관계 행렬 (Factor Correlation Matrix)
# ------------------------------------------------------------------
def chart_factor_correlation():
    kr_corr = corr_df[corr_df['market'] == 'KR']
    pivot = kr_corr.pivot(index='factor_1', columns='factor_2', values='correlation')
    pivot = pivot.reindex(index=['Mkt-RF', 'SMB', 'HML'], columns=['Mkt-RF', 'SMB', 'HML'])

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=ACADEMIC_COLORS['bg'])
    ax.set_facecolor(ACADEMIC_COLORS['bg'])

    mask = np.triu(np.ones_like(pivot, dtype=bool), k=1)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, linecolor='white',
                cbar_kws={'label': '상관계수'}, ax=ax, mask=mask)

    ax.set_title('한국 팩터 상관관계 행렬', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('')
    ax.set_ylabel('')

    save_fig(fig, '04_factor_correlation_matrix.png')

# ------------------------------------------------------------------
# 5. 60개월 이동 평균 팩터 프리미엄 (Rolling 60-month)
# ------------------------------------------------------------------
def chart_rolling_60m():
    window = 60
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, facecolor=ACADEMIC_COLORS['bg'])

    for idx, (col, color, ax) in enumerate(zip(['Mkt-RF', 'SMB', 'HML'],
                                                [ACADEMIC_COLORS['Mkt-RF'],
                                                 ACADEMIC_COLORS['SMB'],
                                                 ACADEMIC_COLORS['HML']],
                                                axes)):
        rolling_mean = factors[col].rolling(window=window).mean() * 100
        rolling_std = factors[col].rolling(window=window).std() * 100
        rolling_ci = 1.96 * rolling_std / np.sqrt(window)

        ax.fill_between(factors['date'], rolling_mean - rolling_ci, rolling_mean + rolling_ci,
                        color=color, alpha=0.15)
        ax.plot(factors['date'], rolling_mean, color=color, linewidth=1.5, label=f'{col} (60M 이동평균)')
        ax.axhline(0, color='#999999', linestyle='--', linewidth=0.8)
        ax.axhline(rolling_mean.mean(), color=color, linestyle=':', linewidth=1, alpha=0.7)

        ax.set_ylabel(f'{col} (%)', fontsize=11)
        ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=False, edgecolor='#333333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=9)

    axes[0].set_title('60개월 이동 평균 팩터 프리미엄 (±95% 신뢰구간)', fontsize=16, fontweight='bold', pad=15)
    axes[-1].set_xlabel('연도', fontsize=12)

    plt.tight_layout()
    save_fig(fig, '05_rolling_60m_premiums.png')

# ------------------------------------------------------------------
# 6. 알파 분포 (Alpha Distribution Histogram)
# ------------------------------------------------------------------
def chart_alpha_distribution():
    alphas = regression['alpha'] * 100
    t_stats = regression['alpha_t']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=ACADEMIC_COLORS['bg'])

    # 히스토그램
    ax1 = axes[0]
    ax1.set_facecolor(ACADEMIC_COLORS['bg'])
    n_bins, bins, patches = ax1.hist(alphas, bins=12, color=ACADEMIC_COLORS['Mkt-RF'],
                                     edgecolor='white', alpha=0.85)
    ax1.axvline(alphas.mean(), color=ACADEMIC_COLORS['HML'], linestyle='--', linewidth=2,
                label=f'평균: {alphas.mean():.2f}%')
    ax1.axvline(0, color='#333333', linestyle='-', linewidth=1)
    ax1.set_title('25 포트폴리오 알파 분포', fontsize=14, fontweight='bold')
    ax1.set_xlabel('알파 (%)', fontsize=11)
    ax1.set_ylabel('빈도', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # t-통계량 산점도
    ax2 = axes[1]
    ax2.set_facecolor(ACADEMIC_COLORS['bg'])
    colors_t = [ACADEMIC_COLORS['HML'] if abs(t) > 2 else ACADEMIC_COLORS['SMB'] for t in t_stats]
    ax2.scatter(range(len(t_stats)), t_stats, c=colors_t, edgecolor='white', s=120, zorder=3)
    ax2.axhline(2, color=ACADEMIC_COLORS['HML'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(-2, color=ACADEMIC_COLORS['HML'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(0, color='#333333', linestyle='-', linewidth=0.8)
    ax2.set_title('알파 t-통계량 (|t|>2 유의)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('포트폴리오', fontsize=11)
    ax2.set_ylabel('t-통계량', fontsize=11)
    ax2.set_xticks(range(len(t_stats)))
    ax2.set_xticklabels(regression['portfolio'], rotation=45, ha='right', fontsize=7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, '06_alpha_distribution.png')

# ------------------------------------------------------------------
# 7. 베타 분포 (Beta Box Plots)
# ------------------------------------------------------------------
def chart_beta_distribution():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=ACADEMIC_COLORS['bg'])
    ax.set_facecolor(ACADEMIC_COLORS['bg'])

    beta_data = [
        regression['beta_mkt'].values,
        regression['beta_smb'].values,
        regression['beta_hml'].values
    ]
    bp = ax.boxplot(beta_data, labels=['시장 베타 (Mkt-RF)', '규모 베타 (SMB)', '가치 베타 (HML)'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='#333333', linewidth=2),
                    whiskerprops=dict(color='#666666', linewidth=1.2),
                    capprops=dict(color='#666666', linewidth=1.2),
                    flierprops=dict(marker='o', markerfacecolor='#B85C38', markersize=6, alpha=0.7))

    colors = [ACADEMIC_COLORS['Mkt-RF'], ACADEMIC_COLORS['SMB'], ACADEMIC_COLORS['HML']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1)

    ax.axhline(0, color='#999999', linestyle='--', linewidth=0.8)
    ax.axhline(1, color='#999999', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.set_title('25 포트폴리오 베타 분포 (Box Plot)', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('베타 계수', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)

    save_fig(fig, '07_beta_distribution.png')

# ------------------------------------------------------------------
# 8. 미국 vs 한국 비교 (US vs Korea Comparison)
# ------------------------------------------------------------------
def chart_us_vs_korea():
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=ACADEMIC_COLORS['bg'])
    ax.set_facecolor(ACADEMIC_COLORS['bg'])

    factors_list = ['Mkt-RF', 'SMB', 'HML']
    x = np.arange(len(factors_list))
    width = 0.35

    # US 데이터는 퍼센트 단위, KR은 소수 -> 퍼센트로 변환
    us_means = []
    kr_means = []
    for f in factors_list:
        us_row = us_vs_kr[(us_vs_kr['factor'] == f) & (us_vs_kr['market'] == 'US')]
        kr_row = us_vs_kr[(us_vs_kr['factor'] == f) & (us_vs_kr['market'] == 'KR')]
        us_val = us_row['mean'].values[0]
        kr_val = kr_row['mean'].values[0] * 100
        # US 값이 이미 퍼센트인지 확인 (std가 4.5 정도면 퍼센트)
        if us_row['std'].values[0] > 1:
            us_val = us_val  # 이미 퍼센트
        else:
            us_val = us_val * 100
        us_means.append(us_val)
        kr_means.append(kr_val)

    bars1 = ax.bar(x - width/2, us_means, width, label='미국 (1963-1991)',
                   color=ACADEMIC_COLORS['US'], edgecolor='#333333', linewidth=0.8)
    bars2 = ax.bar(x + width/2, kr_means, width, label='한국 (2000-2026)',
                   color=ACADEMIC_COLORS['KR'], edgecolor='#333333', linewidth=0.8)

    # 값 레이블
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

    ax.axhline(0, color='#333333', linewidth=0.8)
    ax.set_title('팩터 프리미엄 비교: 미국 vs 한국', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('월간 평균 수익률 (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(factors_list, fontsize=12)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#333333', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)

    # 주석
    ax.text(0.02, 0.02,
            '※ 미국: Fama-French (1993), 1963.07~1991.12 (342개월)\n'
            '※ 한국: 본 연구, 2000.07~2026.05 (311개월)',
            transform=ax.transAxes, fontsize=8, color='#666666',
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    save_fig(fig, '08_us_vs_korea_comparison.png')

# ------------------------------------------------------------------
# 메인 실행
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("한국 Fama-French 3-Factor v2 시각화 생성")
    print("=" * 60)

    chart_cumulative_returns()
    chart_factor_premium()
    chart_25_heatmap()
    chart_factor_correlation()
    chart_rolling_60m()
    chart_alpha_distribution()
    chart_beta_distribution()
    chart_us_vs_korea()

    print("=" * 60)
    print("모든 시각화 생성 완료!")
    print(f"저장 위치: {CHARTS_DIR}")
    print("=" * 60)
