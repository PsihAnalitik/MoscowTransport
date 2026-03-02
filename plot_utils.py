import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fast_plot(
    data: pd.DataFrame,
    date_col: str = 'date',
    value_col: str = 'value',
    title: str = 'Временной ряд',
    figsize: tuple = (14, 5),
    ma_windows: list = None,
    show_stats: bool = True
):
    """
    Быстрая визуализация временного ряда с базовой статистикой.

    Parameters
    ----------
    data : pd.DataFrame
        Датафрейм с данными.
    date_col : str
        Название колонки с датами.
    value_col : str
        Название колонки со значениями.
    title : str
        Заголовок графика.
    figsize : tuple
        Размер фигуры (ширина, высота).
    ma_windows : list[int] | None
        Окна скользящих средних, например [7, 30].
    show_stats : bool
        Показывать ли блок со статистикой на графике.
    """
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    fig, ax = plt.subplots(figsize=figsize)

    # --- основной ряд ---
    ax.plot(
        df[date_col], df[value_col],
        color='#4C72B0', alpha=0.6, linewidth=1,
        label='Факт'
    )

    # --- скользящие средние ---
    colors_ma = ['#C44E52', '#55A868', '#8172B2']
    if ma_windows is None:
        ma_windows = [7, 30]

    for i, w in enumerate(ma_windows):
        ma = df[value_col].rolling(window=w, center=True).mean()
        ax.plot(
            df[date_col], ma,
            color=colors_ma[i % len(colors_ma)],
            linewidth=2,
            label=f'MA({w})'
        )

    # --- статистика ---
    if show_stats:
        stats_text = (
            f"Период: {df[date_col].min():%Y-%m-%d} → {df[date_col].max():%Y-%m-%d}\n"
            f"Записей: {len(df):,}\n"
            f"Среднее: {df[value_col].mean():.2f}  |  "
            f"Медиана: {df[value_col].median():.2f}\n"
            f"Мин: {df[value_col].min():.2f}  |  "
            f"Макс: {df[value_col].max():.2f}  |  "
            f"Std: {df[value_col].std():.2f}"
        )
        ax.text(
            0.01, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85)
        )

    # --- оформление ---
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Дата', fontsize=11)
    ax.set_ylabel('Значение', fontsize=11)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    sns.despine()
    fig.tight_layout()

    plt.show()