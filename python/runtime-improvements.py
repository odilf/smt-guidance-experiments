import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import sqlite3

    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    return np, pl, plt, sqlite3


@app.cell
def _():
    file = "../../../results-v03.db"
    return (file,)


@app.cell(hide_code=True)
def _(file, pl, sqlite3):
    conn = sqlite3.connect(file)

    # Select a statistic with a name from the key value pairs.
    statistic = (
        lambda name: pl.col("decoded")
        .struct.field("entries")
        .list.eval(
            pl.when(pl.element().list.get(0) == name)
            .then(pl.element().list.get(1))
            .otherwise(None)
        )
        .list.drop_nulls()
        .list.first()
    )

    df = (
        pl.read_database(
            query="""SELECT bench.problem_id, runtime, implementation, tactic_help, sat, bench.statistics
            FROM bench JOIN solution ON solution.problem_id = bench.problem_id""",
            connection=conn,
        )
        .with_columns(runtime=pl.col("runtime") / 1e3)
        .with_columns(
            decoded=pl.col("statistics").str.json_decode(),
        )
        .with_columns(
            memory=statistic("memory").str.to_decimal(),
            z3_time=statistic("time").str.to_decimal(),
            conflicts=statistic("conflicts").str.to_decimal(),
            decisions=statistic("decisions").str.to_decimal(),
        )
        .drop("decoded")
        .drop("statistics")
    )

    df

    print(f"Total data points: {len(df)}")
    print(f"Number of solutions: {df.unique('problem_id').count()}")
    df
    return (df,)


@app.cell
def _(df, pl):
    def filter_bench(df, impl: str, help: float):
        return df.filter(
            (pl.col("implementation") == impl) & (pl.col("tactic_help") == help)
        )


    def compare_col(
        impl: str, help_a: float, help_b: float, column="runtime"
    ) -> (list[int], list[int]):
        problem_ids = (
            df.filter(pl.col("sat") == '"sat"')
            # .select("problem_id")
            .unique("problem_id")
            .sort("problem_id")
        )

        col_a = f"{column}_a"
        col_b = f"{column}_b"

        cmp = (
            problem_ids.join(
                filter_bench(df, impl, help_a),
                on=pl.col("problem_id"),
                suffix="_a",
            )
            .join(
                filter_bench(df, impl, help_b),
                on=pl.col("problem_id"),
                suffix="_b",
            )
            .sort("problem_id")
            .select("problem_id", col_a, col_b)
        )

        relatives = cmp.get_column(col_a) / cmp.get_column(col_b)
        absolutes = cmp.get_column(col_a) - cmp.get_column(col_b)

        return relatives, absolutes


    # columns = ["runtime", "z3_time", "memory", "conflicts", "decisions"]
    columns = ["runtime", "z3_time", "decisions"]


    def _():
        for column in columns:
            z3_relatives, z3_absolutes = compare_col("z3", 0.0, 0.9, column=column)
            nood_relatives, nood_absolutes = compare_col(
                "z3-noodler", 0.0, 0.9, column=column
            )
            print(f"\n--- {column} ---")
            print(
                f"z3_relative = {z3_relatives.mean():.3f} ± {z3_relatives.std():.3f}"
            )
            print(
                f"z3_absolute = {z3_absolutes.mean():.3f} ± {z3_absolutes.std():.3f}"
            )
            print(
                f"noodler_relative = {nood_relatives.mean():.3f} ± {nood_relatives.std():.3f}"
            )
            print(
                f"noodler_absulute = {nood_absolutes.mean():.3f} ± {nood_absolutes.std():.3f}"
            )


    _()
    return (filter_bench,)


@app.cell
def _(df, filter_bench, pl):
    from scipy.stats import ttest_ind


    def ttests(impl: str, help_a: float, help_b: float, column="runtime"):
        problem_ids = (
            df.filter(pl.col("sat") == '"sat"')
            .unique("problem_id")
            .sort("problem_id")
        )

        col_a = f"{column}_a"
        col_b = f"{column}_b"

        cmp = (
            problem_ids.join(
                filter_bench(df, impl, help_a),
                on=pl.col("problem_id"),
                suffix="_a",
            )
            .join(
                filter_bench(df, impl, help_b),
                on=pl.col("problem_id"),
                suffix="_b",
            )
            .sort("problem_id")
            .select("problem_id", col_a, col_b)
        )

        runtime_a = cmp.select(col_a)
        runtime_b = cmp.select(col_b)

        return ttest_ind(runtime_a, runtime_b)


    ttests("z3", 0.0, 0.9).pvalue, ttests("z3-noodler", 0.0, 0.9).pvalue
    return


@app.cell
def _(nood_relatives, np, plt, z3_relatives):
    plt.figure(dpi=300)
    # bin_edges = np.linspace(0, 29, 150)
    bin_edges = np.linspace(0, 10, 110)
    plt.hist(z3_relatives, bin_edges)
    plt.ylabel("count")
    plt.xlabel("speedup")
    plt.xlim(-1, 10)
    plt.savefig("../../paper/assets/plots/z3-relative-speedup.svg")
    plt.show()


    plt.figure(dpi=300)
    plt.hist(nood_relatives, bins=bin_edges)
    plt.ylabel("count")
    plt.xlabel("speedup")
    plt.xlim(-1, 10)
    plt.savefig("../../paper/assets/plots/z3-noodler-relative-speedup.svg")
    plt.show()
    return


@app.cell
def _(nood_absolutes, plt, z3_absolutes):
    plt.hist(z3_absolutes, bins=100)
    plt.show()
    plt.hist(nood_absolutes, bins=100)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
