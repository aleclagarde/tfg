import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from scipy.stats import shapiro


def generate_summary(results, lat_data):
    summary_emissions_data = [["", "", "min", "max", "mean", "median", "std"]]
    summary_latency_data = [["", "", "min", "max", "mean", "median", "std"]]
    summary_data = [["prov", "cpu_m", "cpu_c", "reg", "cntr", "em (g)", "comp (s)", "lat (s)"]]

    mult = 1000
    for result in results:
        new_row = [
            "em (g)",
            result.origin[0],
            result.emissions.min() * mult,
            result.emissions.max() * mult,
            result.emissions.mean() * mult,
            result.emissions.median() * mult,
            result.emissions.std() * mult,
        ]
        summary_emissions_data.append(new_row)

        summary_latency_data.append(
            [
                "latency (s)",
                result.origin[0],
                lat_data[result.origin[0]].min(),
                lat_data[result.origin[0]].max(),
                lat_data[result.origin[0]].mean(),
                lat_data[result.origin[0]].median(),
                lat_data[result.origin[0]].std(),
            ]
        )

        summary_data.append(
            [
                result.origin[0],
                result.cpu_model[0],
                result.cpu_count[0],
                result.region[0],
                result.country_name[0],
                result.emissions.mean() * mult,
                result.duration.mean(),
                lat_data[result.origin[0]].mean(),
            ]
        )

    with open("reports/report_figures/summary_table/summary_emissions.html", "w") as f:
        f.write(tabulate(summary_emissions_data, headers="firstrow", tablefmt="html"))
    with open("reports/report_figures/summary_table/summary_lat.html", "w") as f:
        f.write(tabulate(summary_latency_data, headers="firstrow", tablefmt="html"))
    with open("reports/report_figures/summary_table/summary.html", "w") as f:
        f.write(tabulate(summary_data, headers="firstrow", tablefmt="html"))


def gen_normality_results(results: list[pd.DataFrame]):
    sns.set(style="darkgrid")
    my_pal = {"t5_0": "skyblue", "t5_10": "olive", "t5_20": "gold", "t5_30": "teal"}
    df = pd.concat(results, ignore_index=True)
    violin = sns.violinplot(x="origin", y="emissions", data=df, scale="width", palette=my_pal)
    figure = violin.get_figure()
    figure.set_size_inches(10, 10)
    figure.tight_layout()
    figure.savefig("reports/report_figures/normality/violin_comb.png")

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    sns.histplot(data=results[0], x="emissions", kde=True, color="skyblue", bins=12, ax=axs[0, 0]).set(title="Baseline")
    sns.histplot(data=results[1], x="emissions", kde=True, color="olive", bins=12, ax=axs[0, 1]).set(title="10%")
    sns.histplot(data=results[2], x="emissions", kde=True, color="gold", bins=12, ax=axs[1, 0]).set(title="20%")
    sns.histplot(data=results[3], x="emissions", kde=True, color="teal", bins=12, ax=axs[1, 1]).set(title="30%")

    fig.tight_layout()
    fig.savefig("reports/report_figures/normality/histograms.png")

    sns.set(style="darkgrid")
    fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10))
    sns.violinplot(x="origin", y="emissions", color="skyblue", data=results[0], scale="width", ax=axs2[0, 0])
    sns.violinplot(x="origin", y="emissions", color="olive", data=results[1], scale="width", ax=axs2[0, 1])
    sns.violinplot(x="origin", y="emissions", color="gold", data=results[2], scale="width", ax=axs2[1, 0])
    sns.violinplot(x="origin", y="emissions", color="teal", data=results[3], scale="width", ax=axs2[1, 1])

    fig2.savefig("reports/report_figures/normality/violinplots.png")

    shapiro_results = [
        [
            "",
            "test_stat",
            "p-value",
        ]
    ]
    for dataframe in results:
        shapiro_test = shapiro(dataframe.emissions)
        new_row = [
            dataframe.origin[0],
            shapiro_test.statistic,
            shapiro_test.pvalue,
        ]
        shapiro_results.append(new_row)
    with open("reports/report_figures/normality/shapiro.html", "w") as f:
        f.write(tabulate(shapiro_results, headers="firstrow", tablefmt="html"))
    print(tabulate(shapiro_results, headers="firstrow", tablefmt="fancy_grid"))


t5_0 = "results/T5_0/API_results.csv"
t5_10 = "results/T5_0.1/API_results.csv"
t5_20 = "results/T5_0.2/API_results.csv"
t5_30 = "results/T5_0.3/API_results.csv"

t5_0_lat = "results/T5_0/latency_results.csv"
t5_10_lat = "results/T5_0.1/latency_results.csv"
t5_20_lat = "results/T5_0.2/latency_results.csv"
t5_30_lat = "results/T5_0.3/latency_results.csv"

sns.set(style="darkgrid")
t5_0 = pd.read_csv(t5_0)
t5_10 = pd.read_csv(t5_10)
t5_20 = pd.read_csv(t5_20)
t5_30 = pd.read_csv(t5_30)

t5_0_lat = pd.read_csv(t5_0_lat)
t5_10_lat = pd.read_csv(t5_10_lat)
t5_20_lat = pd.read_csv(t5_20_lat)
t5_30_lat = pd.read_csv(t5_30_lat)

results_lat = {"t5_0": t5_0_lat, "t5_10": t5_10_lat, "t5_20": t5_20_lat, "t5_30": t5_30_lat}

t5_0['origin'] = 't5_0'
t5_10['origin'] = 't5_10'
t5_20['origin'] = 't5_20'
t5_30['origin'] = 't5_30'

generate_summary([t5_0, t5_10, t5_20, t5_30], results_lat)
gen_normality_results([t5_0, t5_10, t5_20, t5_30])
