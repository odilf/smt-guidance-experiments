# SMT guidance experiments

Experiments for my [bachelor thesis](https://github.com/odilf/bachelor-thesis).

The main tool is built with Rust with a, let's say, _pragmatic_ approach (the code is very messy). I think I'll leave it as is for now tho.

The main idea of the research is too see if guidance is less useful for better solvers. We do this here by constructing domain-specific constraints by getting a solution.

All the results are saved in an SQLite database. Then, we load the database in Python (code in `./python`), using Polars in a Marimo notebook. I do vouch for this setup, it was very nice to use.

## Paper

- [Paper](https://github.com/odilf/bachelor-thesis/releases/download/2025-06-25/paper.pdf)
- [Poster](https://github.com/odilf/bachelor-thesis/releases/download/2025-06-25/poster.pdf)
- [Presentation slides](https://github.com/odilf/bachelor-thesis/releases/download/2025-06-25/presentation.pdf)
- Source code for paper, poster and presentation slides: https://github.com/odilf/bachelor-thesis
