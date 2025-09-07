from matplotlib import pyplot as plt
import json
from pandas import Series
from pathlib import Path


# %%


def make_plot(data_path: Path, model_name: str, dataset_name):
    data = json.loads(data_path.read_text())

    mean_cnt = Series(data["mean_count_err_by_thresh"])
    mean_cnt.index = mean_cnt.index.astype(float)
    std_cnt = Series(data["std_dev_count_err_by_thresh"])
    std_cnt.index = std_cnt.index.astype(float)

    fig, ax = plt.subplots()
    mean_cnt.plot(ax=ax, label='Promedio del Error de Conteo')
    std_cnt.plot(ax=ax, label='Dev. Std. del Error de Conteo')

    ax.set_xlabel('Umbral de Score de Confianza')
    ax.set_ylabel('Error de Conteo')
    ax.set_title(f'Estadísticas de Error de Conteo vs. Score de Confianza\n'
                 f'Modelo: {model_name} - evaluado sobre: {dataset_name}')
    ax.legend()
    return fig
# %%


fig1 = make_plot(
    data_path=Path("data/evaluation/eval-v2-sky1-over-sky2/summary.json"),
    model_name="v2-sky1",
    dataset_name="SKY/Dataset2",
)
fig1.show()
# %%
fig2 = make_plot(
    data_path=Path("data/evaluation/eval-v1-over-sky2/summary.json"),
    model_name="v1",
    dataset_name="SKY/Dataset2",
)
fig2.show()
# %%
fig3 = make_plot(
    data_path=Path("data/evaluation/eval-v2-derval-over-mauron/summary.json"),
    model_name="v2-derval",
    dataset_name="ICAERUS/mauron",
)
fig3.show()
# %%
