#![allow(dead_code)]

use clap::{Parser, Subcommand};
use export::export;
use model::bench::{Config, Constraint, Implementation};
use std::{ffi::OsStr, path::PathBuf};
use tracing::Level;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{
    EnvFilter, Layer,
    fmt::{self, format::FmtSpan, writer::MakeWriterExt},
    layer::SubscriberExt as _,
};

mod db;
mod export;
mod model;

#[derive(clap::Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,

    /// The path to the sqlite3 database.
    #[arg(long)]
    db_path: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Populate the database with the `.smt2` files in `dataset_path`, recursively.
    PopulateDb {
        dataset_path: PathBuf,

        /// Only select 1000 benches, for small-scale tests.
        #[arg(long)]
        small: bool,
    },

    /// Get solutions (models and unsat_cores) to construct constraints for
    /// [running](Command::Run) the benchmarks.
    GetSolutions,

    /// Run the benchmark.
    Run {
        help: f32,

        /// The iteration of the benchmark. There can only be one bench
        /// for each solution per implementation, help pair.
        ///
        /// This is to keep the samples consistent and to be able to fill them in
        /// and continue if the bench is stopped at any point for any reason.
        #[arg(long, short)]
        iteration: u16,
    },

    /// Export the benchmark data as CBOR
    Export {
        /// The path to save the file to.
        output: PathBuf,
    },

    /// Calculate the help of a constraint given a solution
    CalculateHelp {
        solution: String,
        constraint: Constraint,
    },
}

fn main() -> anyhow::Result<()> {
    let file_appender = RollingFileAppender::new(
        Rotation::HOURLY,
        std::env::home_dir().unwrap().join(".cache/z3-experiments/"),
        "log",
    );

    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(
                fmt::Layer::default()
                    .with_writer(file_appender.with_max_level(Level::TRACE))
                    .json(),
            )
            .with(
                fmt::Layer::default()
                    .with_writer(std::io::stdout)
                    .with_span_events(FmtSpan::ACTIVE)
                    .with_filter(EnvFilter::from_default_env()),
            ),
    )
    .expect("Unable to set global tracing subscriber");

    tracing::info!("Started tracing");

    let args = Args::parse();

    let implementation = Implementation::from_env()?;
    tracing::info!(?implementation);

    match args.command {
        Command::PopulateDb {
            dataset_path,
            small,
        } => {
            let mut conn = db::init(args.db_path)?;
            db::populate(&mut conn, &dataset_path, small)?;
        }

        Command::GetSolutions => {
            let mut conn = db::init(args.db_path)?;
            let total = db::get_number_of_problems(&mut conn)?;
            if total == 0 {
                tracing::warn!(
                    "Problem table size is 0, you probably need to run `populate-db` with a valid `--dataset-path`."
                );
            } else {
                tracing::info!("There are {total} problems");
            }

            let span = tracing::info_span!("init z3").entered();
            let ctx = z3::Context::new(&Config::gen_model().z3());
            span.exit();

            let mut problems = db::iter_unsolved_problems(implementation);
            let mut index = 0;
            while let Some(problem) = problems.next(&mut conn)? {
                let _span = tracing::debug_span!("solving problem", ?problem.id).entered();

                if index % 1000 == 0 {
                    tracing::info!(?index);
                }

                tracing::trace!(?problem);
                let solver = z3::Solver::new(&ctx);
                let solution = match problem.solve(&solver, implementation) {
                    Ok(s) => s,
                    Err(err) => {
                        tracing::error!(err = err.to_string());
                        // FIXME: Huge bodge.
                        if !err
                            .to_string()
                            .contains("No arrow in line of repr of model.")
                        {
                            return Err(err);
                        }
                        continue;
                    }
                };

                tracing::trace!(?solution);
                db::insert_solution(&mut conn, &solution, &problem)?;
                drop(solver);

                index += 1;
            }

            if index == 0 {
                tracing::warn!("No more problems left!");
            } else {
                tracing::trace!("Solved {index} problems")
            }
        }

        Command::Run { help, iteration } => {
            let span = tracing::info_span!("init z3").entered();
            let cfg = Config::benchmark();
            // let ctx = z3::Context::new(&cfg.z3());
            span.exit();

            let mut conn = db::init(args.db_path)?;
            let mut problems = db::iter_unbenched_problems(implementation, help, iteration);

            let mut index = 0;
            let mut unfinished_threads = 0;
            while let Some((problem, solution)) = problems.next(&mut conn)? {
                index += 1;
                let _span =
                    tracing::debug_span!("benching problem", ?problem.id, ?iteration, ?help, ?implementation)
                        .entered();

                let Some(constraints) = solution.sat.generate_constraints(help, problem.hash)
                else {
                    tracing::warn!("Problem has no known sat model, skipping",);
                    continue;
                };

                let bounds_display = constraints
                    .bounds
                    .iter()
                    .map(|(var, value, bound)| bound.to_string(var, value))
                    .enumerate()
                    .fold(String::new(), |mut acc, (i, item)| {
                        if i > 0 {
                            acc.push('\n');
                        }
                        acc.push_str(&item);
                        acc
                    });

                tracing::debug!(?solution.sat, ?constraints.bounds, %bounds_display);

                let (bench, sat, is_thread_finished) = problem.bench_with_constraints(
                    cfg.clone(),
                    constraints,
                    implementation,
                    &cfg,
                    iteration,
                )?;

                if !is_thread_finished {
                    unfinished_threads += 1;
                    if unfinished_threads > 1 {
                        panic!("Too many unfinished threads! ({unfinished_threads})");
                    }
                    tracing::error!(?problem.id, "unfinished thread!");
                }

                assert!(solution.sat.is_sat());
                if sat == z3::SatResult::Unsat {
                    tracing::error!(
                        ?bench.problem_id,
                        ?bench.constraints,
                        "Benchmark found unsat result when original was sat!"
                    );

                    panic!();
                }

                tracing::trace!(?problem, ?solution, ?bench);

                db::insert_bench(&mut conn, &bench, &problem)?;
            }

            if index == 0 {
                tracing::warn!("No more problems left!");
            } else {
                tracing::trace!("Solved {index} problems")
            }
        }

        Command::Export { output } => {
            let ext = output.extension().unwrap_or(OsStr::new(""));
            if ext != "cbor" {
                anyhow::bail!("Extension should be cbor (is .{})", ext.to_string_lossy())
            }

            let mut conn = db::init(args.db_path)?;
            export(&output, &mut conn)?;
        }

        Command::CalculateHelp {
            solution,
            constraint,
        } => {
            println!("{}", constraint.help_given_solution(&solution));
        }
    }

    Ok(())
}
