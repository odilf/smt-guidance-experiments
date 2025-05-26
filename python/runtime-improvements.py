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
    file = "../../../results-v05.db"
    return (file,)


@app.cell
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
            query="""SELECT bench.problem_id, runtime, bench.implementation, tactic_help, sat, bench.statistics
            FROM bench JOIN solution ON solution.problem_id = bench.problem_id AND solution.implementation = bench.implementation""",
            connection=conn,
        )
        .with_columns(runtime=pl.col("runtime") / 1e3)
        .with_columns(
            decoded=pl.col("statistics").str.json_decode(),
        )
        .with_columns(
            memory=statistic("memory").str.to_decimal().cast(pl.Float64),
            z3_time=statistic("time").str.to_decimal().cast(pl.Float64),
            conflicts=statistic("conflicts").str.to_decimal().cast(pl.Float64),
            decisions=statistic("decisions").str.to_decimal().cast(pl.Float64),
        )
        .drop("decoded")
        .drop("statistics")
    )

    df

    print(f"Total data points: {len(df)}")
    print(f"Number of solutions: {df.unique('problem_id').count().get_column("problem_id")[0]}")
    df
    return (df,)


@app.cell
def _(col_af, df, pl):
    def filter_bench(df, impl: str, help: float):
        return df.filter(
            (pl.col("implementation") == impl) & (pl.col("tactic_help") == help)
        )

    # TODO: God-awful name
    def get_compare_col(
        impl: str, help_a: float, help_b: float, column="runtime"
    ) -> (list[int], list[int]):
        problem_ids = (
            df.filter(pl.col("sat") == "sat")
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
            .select("problem_id", col_af, col_b)
        )

        return cmp.get_column(col_a), cmp.get_column(col_b)

    def compare_col(
        impl: str, help_a: float, help_b: float, column="runtime"
    ) -> (list[int], list[int]):
        a, b = get_compare_col(impl, help_a, help_b)

        relatives = b / a
        absolutes = a - b

        return relatives, absolutes


    # columns = ["runtime", "z3_time", "memory", "conflicts", "decisions"]
    columns = ["runtime", "memory", "decisions"]


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
    return columns, compare_col, get_compare_col


@app.cell
def _(columns, compare_col, np, plt):
    bin_edges = np.linspace(0, 10, 101) + 0.05

    def plot_comparison_histograms(columns, compare_col_func):
        """
        Create histograms comparing Z3 and Noodler performance across columns.
    
        Parameters:
        - columns: list of column names to iterate over
        - compare_col_func: your compare_col function
        """
    
        # Calculate number of subplots needed
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5 * n_cols), dpi=300)
    
        # Handle case where there's only one column
        if n_cols == 1:
            axes = axes.reshape(1, -1)
    
        for i, column in enumerate(columns):
            # Get your data
            z3_relatives, z3_absolutes = compare_col_func("z3", 0.0, 0.9, column=column)
            nood_relatives, nood_absolutes = compare_col_func("z3-noodler", 0.0, 0.9, column=column)
        
            # Plot relative comparison
            ax_rel = axes[i, 0]
            ax_rel.hist(z3_relatives, alpha=0.7, bins=bin_edges, label=f'Z3 (μ={z3_relatives.mean():.3f}±{z3_relatives.std():.3f})', 
                       color='blue', edgecolor='black')
            ax_rel.hist(nood_relatives, alpha=0.7, bins=bin_edges, label=f'Noodler (μ={nood_relatives.mean():.3f}±{nood_relatives.std():.3f})', 
                       color='red', edgecolor='black')
            ax_rel.set_title(f'{column} - Relative Performance')
            ax_rel.set_xlabel('Relative Values')
            ax_rel.set_ylabel('Frequency')
            ax_rel.legend()
            ax_rel.grid(True, alpha=0.3)

            ax_rel.xaxis.set_major_locator(plt.MultipleLocator(1))
            ax_rel.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        
            # Plot absolute comparison
            ax_abs = axes[i, 1]
            ax_abs.hist(z3_absolutes, alpha=0.7, bins=bin_edges, label=f'Z3 (μ={z3_absolutes.mean():.3f}±{z3_absolutes.std():.3f})', 
                       color='blue', edgecolor='black')
            ax_abs.hist(nood_absolutes, alpha=0.7, bins=bin_edges, label=f'Noodler (μ={nood_absolutes.mean():.3f}±{nood_absolutes.std():.3f})', 
                       color='red', edgecolor='black')
            ax_abs.set_title(f'{column} - Absolute Performance')
            ax_abs.set_xlabel('Absolute Values')
            ax_abs.set_ylabel('Frequency')
            ax_abs.legend()
            ax_abs.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()

    plot_comparison_histograms(columns, compare_col)
    return


@app.cell
def _(columns, get_compare_col, np):
    from scipy.stats import ttest_ind

    def _():
        for i, column in enumerate(columns):
            for impl in ["z3", "z3-noodler"]:
                a, b = get_compare_col(impl, 0.0, 0.9, column=column)
                result = ttest_ind(np.array(a), np.array(b))
                print(f"ttest pvalue for {impl} is {result.pvalue:.3f}")
    _()
    return


if __name__ == "__main__":
    app.run()
