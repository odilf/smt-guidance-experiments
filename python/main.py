import sqlite3

import numpy as np
import polars as pl
import matplotlib.pyplot as plt


def main():
    conn = sqlite3.connect("../results.db")

    df = pl.read_database(
        query="""SELECT bench.problem_id, runtime, implementation, tactic_help, sat
            FROM bench JOIN solution ON solution.problem_id = bench.problem_id""",
        connection=conn,
    ).sort(by=["implementation", "tactic_help"])

    problem_ids = (
        df.filter(pl.col("sat") == '"sat"')
        .select("problem_id")
        .unique("problem_id")
        .sort("problem_id")
    )

    relatives = []
    absolutes = []
    for (problem_id,) in problem_ids.rows():
        runtime0 = df.filter(
            (pl.col("problem_id") == problem_id) & (pl.col("tactic_help") == 0.0)
        ).get_column("runtime")

        if len(runtime0) != 1:
            continue
        runtime0 = runtime0[0]

        runtime05 = df.filter(
            (pl.col("problem_id") == problem_id) & (pl.col("tactic_help") == 0.5)
        ).get_column("runtime")

        if len(runtime05) != 1:
            continue
        runtime05 = runtime05[0]

        absolute = runtime05 - runtime0
        relative = absolute / runtime0

        relatives.append(relative)
        absolutes.append(absolute)

    print(relatives, absolutes)

    plt.hist(relatives)
    plt.show()
    plt.hist(absolutes)
    plt.show()


if __name__ == "__main__":
    main()
