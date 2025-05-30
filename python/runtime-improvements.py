import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sqlite3

    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    from scipy.stats import gmean, gstd, tmean, tstd, linregress

    plt.rc("text", usetex=False)
    plt.rc("font", family="Libertinus Serif")

    plt.rc(
        "mathtext",
        fontset="custom",
        rm="Libertinus Serif",
        it="Libertinus Serif",
        bf="Libertinus Serif",
    )


    def save(name: str, fig=None):
        if fig is None:
            fig = plt.gcf()

        fig.savefig(f"../../paper/assets/plots/{name}", bbox_inches="tight", transparent=True)
    return gmean, gstd, linregress, np, pl, plt, save, sqlite3, tmean, tstd


@app.cell
def _():
    version = "v09-8"
    file = f"../../../results-{version}.db"
    return (file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Loading the data""")
    return


@app.cell
def _(file, pl, sqlite3):
    conn = sqlite3.connect(file)


    # Select a statistic with a name from the key value pairs.
    def statistic(name):
        return (
            pl.col("decoded")
            .struct.field("entries")
            .list.eval(
                pl.when(pl.element().list.get(0) == name)
                .then(pl.element().list.get(1))
                .otherwise(None)
            )
            .list.drop_nulls()
            .list.first()
        )


    # TIMEOUT_TIME = 631107230402000000
    TIMEOUT_TIME = 6311072304020

    df = (
        pl.read_database(
            query=f"""
            SELECT 
                problem.path AS path,
                bench.problem_id, 
                bench.implementation, 
                bench.bounds AS bounds,
                AVG(runtime) as runtime,
                MIN(runtime) as runtime_min,
                MAX(runtime) as runtime_max,
                AVG(runtime*runtime) - AVG(runtime)*AVG(runtime) as runtime_std,
                help, 
                sat, 
                bench.statistics, 
                model,
                COUNT(1) as total_iterations
            FROM bench 
                JOIN solution ON solution.problem_id = bench.problem_id
                JOIN problem ON problem.id = bench.problem_id
            WHERE runtime < {TIMEOUT_TIME / 2}
            GROUP BY bench.problem_id, bench.implementation, help
            --GROUP BY bench.problem_id, bench.implementation, bench.iteration, help
            -- HAVING total_iterations > 1
            """,
            connection=conn,
        )
        .with_columns(
            pl.col("runtime") / 1e3,
            pl.col("runtime_min") / 1e3,
            timed_out=pl.col("runtime") > TIMEOUT_TIME / 2,
            problem_set=pl.col("path").str.split("/").list.get(-2),
        )
        .with_columns(
            pl.col("bounds")
            .str.count_matches(r"\[\"[^\"]*\"")
            .alias("bounds_length"),
            pl.col("model")
            .str.json_decode(
                dtype=pl.Struct({"assignments": pl.List(pl.List(pl.String))})
            )
            .struct.field("assignments")
            .list.eval(pl.element().list.get(1).len())
            .list.mean()
            .alias("solution_length_mean"),
            # ,
            # .list.drop_nulls()
            # .list.first()
        )
        # .filter(pl.col("problem_set") != "20230329-automatark-lu")
    )

    df

    # print(df.dtypes)
    # df.select(pl.col("model").str.json_decode().struct.field("*"))
    return (df,)


@app.cell
def _(df, pl):
    df.group_by(["implementation", "help"]).agg(
        pl.mean("runtime").alias("mean_runtime"),
        pl.mean("runtime_min").alias("mean_runtime_min"),
        pl.mean("runtime_max").alias("mean_runtime_max"),
        pl.mean("runtime_std").alias("mean_runtime_std"),
        pl.sum("timed_out").alias("timeouts"),
        pl.max("problem_id").alias("last_problem_solved"),
        pl.sum("total_iterations"),
    ).sort(["implementation", "help"])
    return


@app.cell(hide_code=True)
def _():
    # .with_columns(
    #     decoded=pl.col("statistics").str.json_decode(),
    # )
    # .with_columns(
    #     memory=statistic("memory").str.to_decimal().cast(pl.Float64),
    #     z3_time=statistic("time").str.to_decimal().cast(pl.Float64),
    #     z3_total_time=statistic("total_time")
    #     .str.to_decimal()
    #     .cast(pl.Float64),
    #     conflicts=statistic("conflicts").str.to_decimal().cast(pl.Float64),
    #     decisions=statistic("decisions").str.to_decimal().cast(pl.Float64),
    # )
    # .drop("decoded")
    # .drop("statistics")
    return


@app.cell(hide_code=True)
def _(df, pl):
    def problem_sets():
        return df.group_by(["problem_set"]).agg([pl.count("problem_id")])


    problem_sets()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Summary of results""")
    return


@app.cell(hide_code=True)
def _(
    absolute_difference,
    columns,
    compare_col,
    gmean,
    gstd,
    helps,
    impls,
    mo,
    pl,
    relative_speedup,
    tmean,
    tstd,
):
    from collections import defaultdict

    summary = defaultdict(lambda: defaultdict(dict))

    stats = [
        ("rel", relative_speedup, gmean, gstd),
        (
            "rel_weighted",
            compare_col,
            lambda data: gmean(data[0] / data[1], weights=data[0]),
            lambda data: gstd(data[0] / data[1]),
        ),
        (
            "rel_1-100",
            lambda impl, ha, hb, col: compare_col(
                impl,
                ha,
                hb,
                col,
                filter=(pl.col("runtime") > 1) & (pl.col("runtime") < 100),
            ),
            lambda data: gmean(data[0] / data[1], weights=data[0]),
            lambda data: gstd(data[0] / data[1]),
        ),
        (
            "rel_sub1ms",
            lambda impl, ha, hb, col: compare_col(
                impl, ha, hb, col, filter=(pl.col("runtime") < 1)
            ),
            lambda data: gmean(data[0] / data[1], weights=data[0]),
            lambda data: gstd(data[0] / data[1]),
        ),
        ("rel_arithmetic", relative_speedup, tmean, tstd),
        ("abs", absolute_difference, tmean, tstd),
    ]


    def _():
        for column in columns:
            for impl in impls:
                for statname, f, mean, std in stats:
                    data = f(impl, helps[0], helps[1], column)
                    m = float(mean(data))
                    sd = float(std(data))
                    summary[column][impl][statname] = (m, sd)
                
    _()

    def show_summary():
        output = """
            ## Summary Table
        
            | Implementation | Column | Statistic | Mean | Std |
            |---|---|---|---:|---:|
        """.lstrip()

        for column in columns:
            for statname, *_ in stats:
                for impl in impls:
                    mean, std = summary[column][impl][statname]
                    output += f"| **{impl}** | {column} | {statname} | **{mean:.2f}** | {std:.2f} \n"
        return output


    mo.md(show_summary())
    return (summary,)


@app.cell
def _(summary):
    import toml

    with open("../../paper/assets/plots/summary.toml", "w") as f:
        toml.dump(summary, f)
    return


@app.cell(hide_code=True)
def _(compare_col, df, helps, mo, pl):
    mo.md(
        rf"""
    - Total data points: **{len(df)}**
    - Number of problems benched: {df.unique("problem_id").count().get_column("problem_id")[0]}
    - Number of Z3str3 data points: {df.filter(pl.col("implementation") == "z3str3").count().get_column("problem_id")[0]}
    - Number of Z3-noodler data points: {df.filter(pl.col("implementation") == "z3-noodler").count().get_column("problem_id")[0]}
    - {compare_col("z3str3", helps[0], helps[1], "runtime")[0].count()=}
    - {compare_col("z3-noodler", helps[0], helps[1], "runtime")[0].count()=}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <br>
    <br>
    <br>

    # Plots
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Global variables""")
    return


@app.cell
def _():
    columns = ["runtime", "runtime_min"]
    descriptions = ["average", "best case", "worst case"]
    helps = [0.0, 0.9]
    impls = ["z3str3", "z3-noodler"]
    units = ["ms", "ms", "ms"]
    column_relative_x_axis = [
        "problems/s increase",
        "problems/s increase",
        "problems/mb increase",
        "problems/decision increase",
    ]
    return column_relative_x_axis, columns, descriptions, helps, impls


@app.cell
def _(plt):
    import matplotlib.colors as mcolors


    def impl_color(impl: str):
        match impl:
            case "z3str3":
                return (*mcolors.to_rgb("#00da69"), 1.0)
            case "z3-noodler":
                return (*mcolors.to_rgb("#ac3fff"), 1.0)
            case _:
                raise Exception(f"Uknown impl: {impl}")


    def impl_cmap(impl: str):
        match impl:
            case "z3str3":
                cmap = plt.get_cmap("Greens")
            case "z3-noodler":
                cmap = plt.get_cmap("Purples")
            case _:
                raise Exception(f"Uknown impl: {impl}")
        # cmap.set_bad("k")
        return cmap


    def impl_marker(impl: str):
        match impl:
            case "z3str3":
                return "o"
            case "z3-noodler":
                return "D"
            case _:
                raise Exception(f"Uknown impl: {impl}")


    def other_impl(impl: str):
        match impl:
            case "z3str3":
                return "z3-noodler"
            case "z3-noodler":
                return "z3str3"
            case _:
                raise Exception(f"Uknown impl: {impl}")
    return impl_cmap, impl_color, impl_marker


@app.cell
def _(df, pl):
    def filter_bench(df, impl: str, help: float):
        return df.filter(
            (pl.col("implementation") == impl) & (pl.col("help") == help)
        )


    # TODO: Handle iteration column.
    def compare_col(
        impl: str,
        help_a: float,
        help_b: float,
        column,
        filter=True,
    ) -> (pl.Series, pl.Series):
        """
        Finds all shared problems that (impl, help_a) and (impl, help_b) have benched, and returns
        a specific column.

        You can optionally pass a `filter` to filter the columns.
        """

        problem_ids = (
            df.filter(pl.col("sat") == "sat")
            .unique("problem_id")
            .sort("problem_id")
        )

        col_a = f"{column}_a"
        col_b = f"{column}_b"

        cmp = (
            problem_ids.join(
                filter_bench(df, impl, help_a),
                on="problem_id",
                suffix="_a",
            )
            .join(
                filter_bench(df, impl, help_b),
                on="problem_id",
                suffix="_b",
            )
            # .join(
            #     filter_bench(df, other_impl(impl), help_a),
            #     on="problem_id",
            #     suffix="_other_a",
            # )
            # .join(
            #     filter_bench(df, other_impl(impl), help_b),
            #     on="problem_id",
            #     suffix="_other_b",
            # )
            .filter(filter)
            .sort("problem_id")
            .select("problem_id", col_a, col_b)
        )

        return cmp.get_column(col_a), cmp.get_column(col_b)


    def compare_col_map(
        f, impl: str, help_a: float, help_b: float, column="runtime", filter=True
    ) -> (list[int], list[int]):
        a, b = compare_col(impl, help_a, help_b, column=column, filter=filter)
        return f(a, b)


    def gen_col_map(f):
        return lambda impl, help_a, help_b, column, filter=True: compare_col_map(
            f, impl, help_a, help_b, column, filter
        )


    absolute_difference = gen_col_map(lambda a, b: b - a)
    relative_speedup = gen_col_map(lambda a, b: a / b)
    return absolute_difference, compare_col, relative_speedup


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Histograms""")
    return


@app.cell(hide_code=True)
def _(column_relative_x_axis, descriptions, impl_color, np, plt):
    def plot_comparison_histograms(
        columns, compare_col_func, edgecolor="none", linewidth=1
    ):
        """
        Create histograms comparing Z3 and Noodler performance across columns.
        """

        # Calculate number of subplots needed
        n_cols = len(columns)
        fig, axes = plt.subplots(
            n_cols, 2, figsize=(10, min(3 * n_cols, 8)), dpi=300
        )

        # Handle case where there's only one column
        if n_cols == 1:
            axes = axes.reshape(1, -1)

        for i, (column, x_label, description) in enumerate(
            zip(columns, column_relative_x_axis, descriptions)
        ):
            # Get your data
            z3_relatives, z3_absolutes = compare_col_func("z3str3", column)
            nood_relatives, nood_absolutes = compare_col_func("z3-noodler", column)

            # Plot relative comparison
            # bin_edges = np.logspace(-1, 1, 101) + 0.05
            bin_edges = np.logspace(
                np.log(min(z3_relatives.min(), nood_relatives.min())),
                np.log(max(z3_relatives.max(), nood_relatives.max())),
                101,
            )
            ax_rel = axes[i, 0]
            ax_rel.axvspan(0, 1, alpha=0.1, color="red")
            ax_rel.hist(
                z3_relatives,
                bins=bin_edges,
                label=f"Z3str3",
                color=impl_color("z3str3"),
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
            ax_rel.hist(
                nood_relatives,
                bins=bin_edges,
                label="Z3-Noodler",
                color=impl_color("z3-noodler"),
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
            ax_rel.set_title(f"Relative speedup ({description})")
            ax_rel.set_xlabel("Speedup (problems/s %)")
            ax_rel.set_xscale("log")
            ax_rel.set_ylabel("Frequency")
            ax_rel.set_yscale("log")
            ax_rel.legend()
            ax_rel.grid(True, alpha=0.1)

            # Plot absolute comparison
            z3_mean = z3_absolutes.mean()
            z3_std = z3_absolutes.std()
            nood_mean = nood_absolutes.mean()
            nood_std = nood_absolutes.std()

            std_spread = 0.5
            spread = max(nood_std, z3_std) * std_spread
            xmin = -spread
            xmax = spread

            bin_edges = np.linspace(xmin, xmax, 100)

            ax_abs = axes[i, 1]
            ax_abs.axvspan(0, xmax, alpha=0.1, color="red")
            ax_abs.hist(
                z3_absolutes,
                bins=bin_edges,
                label=f"Z3str3",
                # label=f"Z3str3 (μ={z3_absolutes.mean():.2f}±{z3_absolutes.std():.2f} s/problem)",
                color=impl_color("z3str3"),
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
            ax_abs.hist(
                nood_absolutes,
                bins=bin_edges,
                label=f"Z3-Noodler",
                # label=f"Z3-Noodler (μ={nood_absolutes.mean():.2f}±{nood_absolutes.std():.2f} s/problem)",
                color=impl_color("z3-noodler"),
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
            ax_abs.set_title(f"Absolute difference ({description})")
            # ax_abs.set_xlim(xmin, xmax)
            # ax_abs.set_xscale("symlog")
            ax_abs.set_xlabel("Runtime increase (s/problem)")
            ax_abs.set_ylabel("Frequency")
            ax_abs.set_yscale("log")
            ax_abs.legend()
            ax_abs.grid(True, alpha=0.1)

        plt.tight_layout()
        return fig, axes
    return (plot_comparison_histograms,)


@app.cell(hide_code=True)
def _(
    absolute_difference,
    columns,
    helps,
    plot_comparison_histograms,
    plt,
    relative_speedup,
    save,
):
    _fig, _ = plot_comparison_histograms(
        columns,
        lambda impl, column: (
            relative_speedup(impl, helps[0], helps[1], column),
            absolute_difference(impl, helps[0], helps[1], column),
        ),
        edgecolor="black",
        linewidth=0.2,
    )

    save("absolute-diff-relative-speedup-histograms-all-columns.svg", _fig)
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    compare_col,
    gmean,
    gstd,
    helps,
    impl_color,
    impls,
    np,
    original_runtime_max,
    plt,
    save,
):
    def _(n=30, original_runtime_max=None):
        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8, 5))
        for i, weigh in enumerate([False, True]):
            axs[i, 0].set_ylabel("Frequency")
            for ax, impl in zip(axs[i, :], impls):
                original, new = compare_col(impl, helps[0], helps[1], "runtime")
                if original_runtime_max is not None:
                    new = new.filter(original <= original_runtime_max)
                    original = original.filter(original <= original_runtime_max)

                weights = original if weigh else None

                speedups = original / new
                mean = gmean(speedups, weights=weights)
                std = gstd(speedups)

                bin_edges = np.logspace(
                    np.log10(speedups.min()),
                    np.log10(speedups.max()),
                    n + 1,
                )

                counts, _ = np.histogram(speedups, bin_edges, weights=weights)
                counts = counts / counts.sum()

                ax.axvspan(0, 1, color="red", alpha=0.1)
                ax.bar(
                    (bin_edges[:-1] + bin_edges[1:]) / 2,
                    counts,
                    width=np.diff(bin_edges),
                    color=impl_color(impl),
                    edgecolor="black",
                    linewidth=0.5,
                )

                ax.set_xscale("log")
                ax.set_xlabel("Speedup (problems/ms %)")
                # ax.set_yscale("log")
                ax.set_title(
                    f"{impl}, {'weighted' if weigh else 'unweighted'} ($µ_{{speedup}} = {mean:.3f} \\pm {std:.2f}$)"
                )
                ax.grid(True, alpha=0.1)

        plt.tight_layout()
        save("histograms-separate.svg", fig)
        return fig


    _(original_runtime_max=original_runtime_max.value)
    return


@app.cell(hide_code=True)
def _(df, mo, np):
    original_runtime_max = mo.ui.slider(
        steps=np.flip(
            np.logspace(-3, np.log10(df.get_column("runtime").max()), 100)
        )
    )
    original_runtime_max
    return (original_runtime_max,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Heatmaps""")
    return


@app.cell(hide_code=True)
def _(compare_col, helps, impl_cmap, np, plt, save):
    def _():
        from scipy.stats import binned_statistic_2d

        fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=300)

        def do_heatmap(impl, ax):
            a, b = compare_col(impl, helps[0], helps[1], "runtime")
            x = np.array(a)
            y = np.array(a / b)

            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)

            n_bins = 50

            x_bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
            y_bins = np.linspace(y_min, 10, n_bins + 1)
            # y_bins = np.logspace(np.log10(y_min), np.log10(y_max), n_bins + 1)

            counts, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

            im = ax.pcolormesh(
                x_edges,
                y_edges,
                counts.T,
                norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=counts.max()),
                cmap=impl_cmap(impl),
                shading="flat",
                alpha=1,
            )

            ax.set_title(impl)
            ax.set_xscale("log")
            # ax.set_xlabel("Original runtime (s)")
            # ax.set_yscale("log")
            ax.set_xlabel("Original runtime (ms/problem)")
            ax.axhline(1, x_min, x_max, color="white", linestyle="--", linewidth=1)
            plt.colorbar(im, ax=ax)

            # ax.set_ylabel("Speedup (% of problems/second)")

            # ax.set_aspect('equal', adjustable='box')
            # plt.show()

        axes[0].set_ylabel("Speedup (problems/ms %)")
        do_heatmap("z3str3", axes[0])
        do_heatmap("z3-noodler", axes[1])

        save("speedup-vs-original-heatmaps.png", fig)
        return fig

    _()
    return


@app.cell
def _(compare_col, helps, impl_cmap, np, plt, save, tmean, tstd):
    def _(n_bins=10, scale="linear"):
        from scipy.stats import binned_statistic_2d

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7, 3),
            # sharex=True,
            # sharey=True,
            dpi=300,
            layout="constrained",
        )

        def do_heatmap(impl, ax):
            a, b = compare_col(impl, helps[0], helps[1], "runtime")
            x = np.array(a)
            y = np.array(b)

            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)

            # Make axes equal
            g_min = min(x_min, y_min)
            g_max = max(x_max, y_max)
            x_min = y_min = g_min
            x_max = y_max = g_max

            if scale == "log":
                x_bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
                y_bins = np.logspace(np.log10(y_min), np.log10(y_max), n_bins + 1)
            else:
                x_bins = np.linspace(x_min, x_max, n_bins + 1)
                y_bins = np.linspace(y_min, y_max, n_bins + 1)

            counts, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

            import matplotlib.patches as patches

            triangle = patches.Polygon(
                [
                    (x_min, y_min),
                    (x_min, y_max),
                    (x_max, y_max),
                ],
                closed=True,
                facecolor="red",
                alpha=0.1,
            )
            ax.add_patch(triangle)

            ax.set_aspect("equal")
            im = ax.pcolormesh(
                x_edges,
                y_edges,
                counts.T,
                norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=counts.max()),
                # norm=plt.matplotlib.colors.Normalize(vmin=100, vmax=54),
                cmap=impl_cmap(impl),
                shading="flat",
                alpha=1,
            )

            ax.set_title(
                f"{impl} ($µ_{{diff}}={tmean(y - x):.2f} ± {tstd(y - x):.0f} \\, ms$)"
            )
            ax.set_xscale(scale)
            ax.set_xlabel("Original runtime (ms/problem)")
            ax.set_yscale(scale)
            ax.set_aspect(1)

            plt.colorbar(im, ax=ax)
            ax.axline(
                [x_min, y_min],
                [x_max, y_max],
                color="black",
                linestyle="--",
                linewidth=0.5,
            )

        axes[0].set_ylabel("New runtime (ms)")
        do_heatmap("z3str3", axes[0])
        do_heatmap("z3-noodler", axes[1])

        save("new-vs-original-heatmaps.png", fig)
        save("new-vs-original-heatmaps.svg", fig)
        return plt.gca()


    _(n_bins=50, scale="log")
    return


@app.cell(hide_code=True)
def _(compare_col, helps, impl_color, impl_marker, impls, plt):
    def _():
        plt.figure(dpi=400)
        g_max = 0
        for impl in impls:
            original, new = compare_col(impl, helps[0], helps[1], "runtime")
            original = original / 1e3
            new = new / 1e3
            x_min = y_min = 0
            x_max = original.max()
            y_max = new.max()
            g_max = max(x_max, y_max, g_max)

            plt.scatter(
                original,
                new,
                label=impl,
                color=impl_color(impl),
                marker=impl_marker(impl),
                alpha=0.7,
                s=5,
            )

        plt.plot([0, g_max], [0, g_max], color="black", linestyle="--")
        import matplotlib.patches as patches

        triangle = patches.Polygon(
            [
                (0, 0),
                (0, g_max),
                (g_max, g_max),
            ],
            closed=True,
            facecolor="red",
            alpha=0.1,
        )

        plt.gca().add_patch(triangle)

        plt.xlabel("Original runtime (s)")
        plt.ylabel("New runtime (s)")
        plt.legend(loc="upper left")
        return plt.gca()


    _()
    return


@app.cell
def _(
    compare_col,
    helps,
    impl_color,
    impl_marker,
    impls,
    linregress,
    np,
    plt,
    save,
):
    def _(x_max=None, x_min=None, xscale="linear"):
        plt.figure(dpi=900, figsize=(8, 4))
        plt.axhspan(0, 1, color="red", alpha=0.1)
        g_max = 0
        for impl in impls:
            original, new = compare_col(impl, helps[0], helps[1], "runtime")
            x = (original / 1e3).to_numpy()
            y = (original / new).to_numpy()

            x_min = 0 if x_min is None else x_min
            x_max = x.max() if x_max is None else x_max
            y_min = 0
            y_max = y.max()

            y = y[x > x_min]
            x = x[x > x_min]
            y = y[x < x_max]
            x = x[x < x_max]
            g_max = max(x_max, y_max, g_max)

            regress = linregress(x, np.log(y), alternative="greater")

            plt.xscale(xscale)
            plt.yscale("log")
            plt.scatter(
                x,
                y,
                label=f"{impl} (growth rate = {np.exp(regress.slope):.3f} $\\pm$ {regress.stderr:.3f}, $R^2 = {regress.rvalue**2:.2e}$)",
                color=impl_color(impl),
                marker=impl_marker(impl),
                alpha=1,
                s=5,
            )

            if xscale == "linear" and impl == "z3str3":
                f = lambda x: np.exp(regress.intercept + regress.slope * x)

                plt.plot(
                    x,
                    f(x),
                    color=impl_color(impl),
                    alpha=0.1,
                )

        plt.xlabel("Original runtime (s)")
        plt.ylabel("Speedup (problems/s %)")
        # plt.xlim(0, x_max)
        # plt.ylim(0, y_max)
        plt.legend(loc="lower right")
        save(f"speedup-vs-original-{xscale}.svg")
        return plt.gca()


    (
        _(),
        # _(x_min=x_min.value, x_max=x_max.value, xscale="log"),
    )
    return


@app.cell(hide_code=True)
def _(compare_col, helps, impl_cmap, impl_color, impls, np, plt):
    def _(n):
        from scipy.spatial.distance import cdist

        # Generate sample data with both dense and sparse regions
        np.random.seed(42)

        g_max = 0
        for impl in impls:
            # Combine all data
            x, y = compare_col(impl, helps[0], helps[1], "runtime")
            x = x.to_numpy() / 1e3
            y = y.to_numpy() / 1e3
            y = x / y

            max = np.max([x, y])
            g_max = np.max([max, g_max])

            # Alternative approach: Threshold-based hexbin filtering
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

            # Create hexbin plot and get the bin counts
            hb = ax.hexbin(
                x, y, gridsize=n, cmap="Blues", alpha=0.0
            )  # Invisible first pass
            counts = hb.get_array()
            verts = hb.get_offsets()

            # Clear the invisible hexbin
            # ax.clear()

            # Set threshold for "dense" vs "sparse"
            density_threshold = 10

            # Create the actual hexbin but only show bins above threshold
            hb_dense = ax.hexbin(
                x,
                y,
                gridsize=n,
                cmap=impl_cmap(impl),
                alpha=1.0,
                mincnt=density_threshold,
                norm=plt.matplotlib.colors.LogNorm(),
            )

            # For points in low-density hexagons, show as individual scatter points
            # This is approximate - for exact control you'd need to map each point to its hexagon
            from matplotlib.collections import RegularPolyCollection
            import matplotlib.patches as patches

            # Create mask for points in sparse hexagons (approximation using local density)
            def local_density_mask(x, y, radius, threshold):
                """Approximate sparse areas using local point density"""
                points = np.column_stack([x, y])
                distances = cdist(points, points)
                local_counts = np.sum(distances < radius, axis=1)
                return local_counts < threshold

            sparse_approx = local_density_mask(
                x, y, radius=max / n, threshold=density_threshold
            )
            ax.scatter(
                x[sparse_approx],
                y[sparse_approx],
                s=40,
                color=impl_color(impl),
                alpha=0.9,
                linewidth=1,
                zorder=5,
                label=impl,
            )
            ax.set_yscale("log")
            plt.show()

        ax.set_title(
            "Hexbin for Dense Areas + Scatter for Sparse Areas", fontsize=14
        )
        ax.set_xlabel("Runtime without help (s)")
        ax.set_ylabel("Runtime with help (s)")
        ax.legend()
        plt.colorbar(hb_dense, label="Point Count in Hexagon")

        plt.show()


    # _(n=25)
    return


@app.cell(hide_code=True)
def _(compare_col, gmean, helps, impl_color, np, plt, save):
    def _(show_range: bool, n: int, column: str):
        fig = plt.figure(dpi=300, figsize=(7, 3))

        def get_xy(impl):
            a, b = compare_col(impl, helps[0], helps[1], column)
            x_all = np.array(a) / 1e3
            y_all = np.array(a / b)
            return x_all, y_all

        xy_z3str3 = get_xy("z3str3")
        xy_z3noodler = get_xy("z3-noodler")

        x_min, x_max = (
            min(xy_z3str3[0].min(), xy_z3noodler[0].min()),
            max(xy_z3str3[0].max(), xy_z3noodler[0].max()),
        )

        def average_runtime_increases(impl, x_all, y_all, offset_factor, n):
            # x_bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), n + 1)
            # plt.xscale("log")
            x_bin_edges = np.linspace(x_min, x_max, n + 1)
            bin_width = np.diff(x_bin_edges)[0]

            x = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
            x += offset_factor * bin_width * 0.15

            y_mean = []
            y_minmax = []
            y_count = []
            for low, high in zip(x_bin_edges, x_bin_edges[1:]):
                samples = y_all[(x_all >= low) & (x_all < high)]
                y_count.append(len(samples))
                if len(samples) < 1:
                    y_mean.append(float("nan"))
                    y_minmax.append([float("nan"), float("nan")])
                else:
                    y_mean.append(gmean(samples))
                    y_minmax.append([min(samples), max(samples)])

            plt.xlabel("Original runtime (s)")
            plt.ylabel("Mean speedup (problems/s %)")
            plt.yscale("log")

            color = impl_color(impl)
            plt.axhspan(0.0, 1.0, alpha=0.1, color="red")
            if not show_range:
                plt.bar(
                    x,
                    y_mean,
                    width=np.diff(x_bin_edges),
                    label=impl,
                    color=color,
                    edgecolor="black",
                )

            else:
                y_minmax = np.array(y_minmax)
                plt.errorbar(
                    x,
                    y_mean,
                    yerr=np.array(
                        [
                            np.abs(y_minmax[:, 0] - y_mean),
                            np.abs(y_minmax[:, 1] - y_mean),
                        ]
                    ),
                    # yerr=y_mean * 2,
                    fmt="o",
                    label=impl,
                    alpha=1,
                    capsize=5,
                    color=color,
                )

                # plt.bar(x, y_count, width=bin_width, label=impl, color='gray')

        if show_range:
            bin_edges = np.linspace(x_min, x_max, n + 1)
            for edge in bin_edges:
                plt.axvline(
                    x=edge, color="gray", linestyle="--", alpha=0.3, linewidth=0.5
                )

        average_runtime_increases("z3str3", *xy_z3str3, offset_factor=-1, n=n)
        average_runtime_increases(
            "z3-noodler", *xy_z3noodler, offset_factor=1, n=n
        )

        plt.legend()
        save(
            "mean-speedup-vs-original-time.svg"
            if not show_range
            else "mean-speedup-vs-original-time-with-range.svg",
            fig,
        )
        return plt.gca()


    _(show_range=True, n=15, column="runtime")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Speedup vs size of solution""")
    return


@app.cell(hide_code=True)
def _(
    compare_col,
    helps,
    impl_color,
    impl_marker,
    impls,
    linregress,
    np,
    plt,
    relative_speedup,
):
    def _():
        plt.figure(dpi=300, figsize=(7, 3))
        for impl in impls:
            x, _ = compare_col(impl, helps[0], helps[1], "solution_length_mean")
            y = relative_speedup(impl, helps[0], helps[1], "runtime")

            regress = linregress(x, np.log(y))
            plt.scatter(
                x,
                y,
                label=f"{impl} (growth rate={np.exp(regress.slope):.3f} $\\pm$ {regress.stderr:.3f})",
                color=impl_color(impl),
                marker=impl_marker(impl),
                alpha=0.1,
            )

            x = np.logspace(np.log10(x.filter(x > 0).min()), np.log10(x.max()), 100)
            plt.plot(x, np.exp(regress.intercept + regress.slope * x), color=impl_color(impl), alpha=1)

        plt.axhspan(0, 1, color="red", alpha=0.1)
        plt.xscale("log")
        plt.xlabel("Size of solution")
        plt.yscale("log")
        plt.ylabel("Speedup (problems/ms %)")
        plt.legend()
        return plt.gca()


    _()
    return


@app.cell
def _(compare_col, helps, impls, np):
    for impl in impls:
        a, b = compare_col(impl, helps[0], helps[1], column="runtime")
        worse = len(np.where(a < b)[0])
        total = len(a)
        print(f"Slowdown in {worse / total * 100:.3f}% of cases for {impl}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Speedup vs help""")
    return


@app.cell(hide_code=True)
def _(df, impl_color, impl_marker, impls, linregress, pl, plt):
    def _():
        plt.figure(dpi=300)
        for impl in impls:
            entries = df.filter(pl.col("help") > 0.0).select(["runtime", "help"])
            x = entries["help"]
            y = entries["runtime"]
            regress = linregress(x, y)
            plt.scatter(
                x,
                y,
                label=f"{impl} (slope={regress.slope:.3f} $\\pm$ {regress.stderr:.3f})",
                color=impl_color(impl),
                marker=impl_marker(impl),
                alpha=0.01,
            )

        plt.legend()
        plt.xlabel("Help")
        plt.yscale("log")
        plt.ylabel("Speedup (problem/ms %)")
        return plt.gca()


    _()
    return


@app.cell(hide_code=True)
def _(columns, gmean, impl_color, impls, np, plt, relative_speedup):
    def _():
        for column in columns:
            plt.figure(dpi=300)
            plt.xlim(0.0, 1.0)
            plt.xlabel("Help")
            plt.yscale("log")
            plt.ylabel("Speedup")
            plt.axhspan(0.0, 1.0, alpha=0.1, color="red")

            helps = np.array([0.0, 0.2, 0.5, 0.98, 0.99])
            for impl in impls:
                speedups = np.array(
                    [
                        gmean(relative_speedup(impl, 0.0, help, column))
                        for help in helps
                    ]
                )
                plt.scatter(helps, speedups, label=impl, color=impl_color(impl))

            plt.legend()
            return plt.gca()


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Speedup vs number of constriants""")
    return


@app.cell(hide_code=True)
def _(df, gmean, helps, impl_color, impls, np, pl, plt, relative_speedup):
    def _():
        plt.figure(dpi=300)
        for impl in impls:
            n_constraints = np.arange(0, df.get_column("bounds_length").max())
            mean_speedups = []
            speedups_minmax = []
            for n in n_constraints:
                speedups = relative_speedup(
                    impl,
                    helps[0],
                    helps[1],
                    "runtime",
                    filter=pl.col("bounds_length_b") == n,
                )
                if len(speedups) == 0:
                    mean_speedups.append(float("nan"))
                    speedups_minmax.append([float("nan"), float("nan")])
                else:
                    mean_speedups.append(float(gmean(speedups)))
                    speedups_minmax.append([min(speedups), max(speedups)])

            speedups_minmax = np.array(speedups_minmax)
            plt.xlabel("Number of constraints")
            plt.yscale("log")
            plt.ylabel("Speedup (problems/ms %)")
            # plt.scatter(n_constraints, mean_speedups)

            plt.errorbar(
                n_constraints,
                mean_speedups,
                # yerr=speedups_minmax.T,
                yerr=np.array(
                    [
                        np.abs(speedups_minmax[:, 0] - mean_speedups),
                        np.abs(speedups_minmax[:, 1] - mean_speedups),
                    ]
                ),
                fmt="o",
                label=impl,
                color=impl_color(impl),
                capsize=5,
            )
            plt.axhspan(0, 1, color="red", alpha=0.1)

        plt.legend()
        return plt.gca()


    _()
    return


@app.cell(hide_code=True)
def _(
    compare_col,
    helps,
    impl_color,
    impl_marker,
    impls,
    linregress,
    np,
    plt,
    relative_speedup,
):
    def _():
        plt.figure(dpi=400)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Speedup')
        plt.xlabel('Number of constraints')
        for impl in impls:
            _, x = compare_col(impl, helps[0], helps[1], "bounds_length")
            y = relative_speedup(impl, helps[0], helps[1], "runtime")
            regress = linregress(x, np.log(y))
            plt.scatter(
                x,
                y,
                label=f"{impl} (growth rate={np.exp(regress.slope):.3f} $\\pm$ {regress.stderr:.3f})",
                color=impl_color(impl),
                marker=impl_marker(impl),
                alpha=0.1,
            )

            x = np.logspace(np.log10(x.filter(x > 0).min()), np.log10(x.max()), 100)
            plt.plot(x, np.exp(regress.intercept + regress.slope * x), color=impl_color(impl), alpha=1)

        plt.legend()
        return plt.gca()
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <br>
    <br>
    <br>
    <br>
    <br>

    # Helper visualizations
    """
    )
    return


@app.cell(hide_code=True)
def _(np, plt, save):
    def _():
        fig = plt.figure(dpi=300, figsize=(4, 2))
        cutoff = 3
        f = lambda lam, x: lam * np.exp(-lam * x)
        plt.vlines(cutoff, 0, f(0.1, cutoff), color="black")
        x = np.linspace(0, 12, 100)
        x_shaded = np.linspace(0, cutoff, 40)
        for lam in [0.1, 1]:
            plt.fill_between(x_shaded, f(lam, x_shaded), alpha=0.1)
            plt.plot(x, f(lam, x), label=f"$\\lambda={lam}$")

        plt.xlim(0, max(x))
        plt.ylim(0, 1)
        plt.legend()
        save("less-than-constraint-lambda-comparison.svg")
        return plt.gca()


    _()
    return


@app.cell(hide_code=True)
def _(np, plt):
    def _():
        x = np.linspace(0, 3, 1000)
        pl = lambda lam, s: np.exp(-lam * s) * (1 - np.exp(-lam))
        plt.figure(dpi=300)
        plt.xlabel("$\\lambda$")
        plt.xscale("log")
        plt.ylabel("$p_\\ell$")
        for size in np.linspace(0, 10, 10):
            cmap = plt.get_cmap("inferno")
            color = cmap(size / 10)
            plt.plot(x, pl(x, size), label=f"|s| = {size}", alpha=1, color=color)
            opt = np.log((size + 1) / size)
            plt.vlines(opt, 0, pl(opt, size), alpha=0.5, color=color)
        plt.legend()
        return plt.gca()


    _()
    return


if __name__ == "__main__":
    app.run()
