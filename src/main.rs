#![allow(dead_code)]

use clap::{Parser, Subcommand};
use export::export;
use model::bench::{Config, Implementation};
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

    /// Get solutions (models and unsat_cores) to construct tactics for
    /// [running](Command::Run) the benchmarks.
    GetSolutions,

    /// Run the benchmark.
    Run {
        implementation: Implementation,
        tactic_help: f32,
    },

    /// Export the benchmark data as CBOR
    Export {
        /// The path to save the file to.
        output: PathBuf,
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
            let solver = z3::Solver::new(&ctx);
            span.exit();

            let mut problems = db::iter_unsolved_problems();
            let mut index = 0;
            while let Some(problem) = problems.next(&mut conn)? {
                let _span = tracing::debug_span!("solving problem", ?problem.id).entered();

                if index % 1000 == 0 {
                    tracing::info!(?index);
                }

                tracing::trace!(?problem);
                let solution = match problem.solve(&solver) {
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
                solver.reset();

                index += 1;
            }
        }

        Command::Run {
            implementation,
            tactic_help,
        } => {
            let span = tracing::info_span!("init z3").entered();
            let cfg = Config::benchmark();
            let ctx = z3::Context::new(&cfg.z3());
            let solver = z3::Solver::new(&ctx);
            span.exit();

            let mut conn = db::init(args.db_path)?;
            let mut problems = db::iter_unbenched_problems(implementation, tactic_help);

            while let Some((problem, solution)) = problems.next(&mut conn)? {
                let _span =
                    tracing::debug_span!("benching problem", ?problem.id, sat=?solution.sat.to_str())
                        .entered();

                let Some(tactic) = solution.sat.generate_tactic(tactic_help, problem.id) else {
                    tracing::warn!("Problem has no known sat model, skipping",);
                    continue;
                };

                tracing::debug!(?solution.sat, ?tactic.bounds);

                let (bench, sat) =
                    problem.bench_with_tactic(&solver, tactic, implementation, &cfg)?;

                if sat != solution.sat.z3() {
                    tracing::error!(
                        ?problem.id,
                        bench_solution=?sat,
                        db_solution=?solution.sat.z3(),
                        "Benchmark got different solution than solution generation!"
                    );

                    continue;
                }

                if matches!(sat, z3::SatResult::Unknown) {
                    tracing::error!("Didn't find solution for");
                    continue;
                }

                tracing::trace!(?problem, ?solution, ?bench);

                db::insert_bench(&mut conn, &bench, &problem)?;
                solver.reset();
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
    }

    Ok(())
}
