use std::{path::PathBuf, sync::mpsc, thread, time::Duration};

use anyhow::Context as _;

use crate::model::solution::Statistics;

use super::{
    Bench, Solution,
    bench::{Config, Constraints, Implementation},
    solution::Sat,
};

const SENTINEL_ID: u32 = 69420;

// TODO: Would be nice to extract the `set-info` parts of the original files.
#[derive(Debug, Clone)]
pub struct Problem {
    pub id: u32,
    pub hash: u64,
    pub path: PathBuf,
    pub content: String,
}

impl Problem {
    pub fn solve(
        &self,
        solver: &z3::Solver,
        implementation: Implementation,
    ) -> anyhow::Result<Solution> {
        solver.from_string(&*self.content);
        let sat = solver.check();

        let sat = match sat {
            z3::SatResult::Sat => {
                let model = solver.get_model().context("Missing model")?;
                Sat::Sat(model.try_into()?)
            }
            z3::SatResult::Unsat => {
                // TODO: Do I need to track what?
                // for assertion in solver.get_assertions() {
                //     solver.assert_and_track(&assertion, p);
                // }
                solver.check_assumptions(&solver.get_assertions());
                let unsat_core = solver.get_unsat_core();
                Sat::Unsat(unsat_core.into())
            }

            z3::SatResult::Unknown => {
                tracing::warn!(reason_unknown=?solver.get_reason_unknown());
                if solver
                    .get_reason_unknown()
                    .is_some_and(|reason| reason.contains("interrupted"))
                {
                    anyhow::bail!("Interrupted.");
                }

                tracing::warn!("Timed out!");

                // TODO: Add `solver.reason_unknown` in metadata
                Sat::Unknown
            }
        };

        let statistics = solver.get_statistics().into();

        Ok(Solution {
            id: SENTINEL_ID,
            implementation,
            sat,
            statistics,
        })
    }

    pub fn bench_with_constraints<'a, 'b>(
        &'a self,
        cfg: Config,
        constraints: Constraints,
        implementation: Implementation,
        configuration: &Config,
        iteration: u16,
    ) -> anyhow::Result<(Bench, z3::SatResult, bool)> {
        // Approximate time start, for database.
        let time_start = jiff::Timestamp::now();

        let (sender, receiver) = mpsc::channel();
        let handle = {
            let content = self.content.clone();
            let constraints = constraints.clone();

            thread::spawn(move || {
                let ctx = z3::Context::new(&cfg.z3());
                let solver = z3::Solver::new(&ctx);
                solver.from_string(content);
                constraints.add_bounds_to_solver(&solver);

                let time_start = jiff::Timestamp::now();
                let sat = solver.check();
                let time_end = jiff::Timestamp::now();
                sender
                    .send((
                        time_end - time_start,
                        sat,
                        solver.get_reason_unknown(),
                        Some(Statistics::from(solver.get_statistics())),
                    ))
                    .unwrap();
            })
        };

        let timeout_time = || jiff::Timestamp::MAX - jiff::Timestamp::MIN;
        let (mut runtime, sat, reason_unknown, statistics, timed_out_thread) =
            match receiver.recv_timeout(cfg.timeout + Duration::from_secs(10)) {
                Ok((runtime, sat, reason_unknown, statistics)) => {
                    (runtime, sat, reason_unknown, statistics, false)
                }
                Err(_) => (
                    timeout_time(),
                    z3::SatResult::Unknown,
                    Some("timeout".to_string()),
                    None,
                    true,
                ),
            };

        tracing::debug!(?runtime);

        if let Some(reason) = reason_unknown {
            if reason.is_empty() || reason == "unknown" {
                // These are fine, I think.
            } else if reason.contains("interrupted") {
                anyhow::bail!("Interrupted.");
            } else if reason.contains("timeout") || reason.contains("canceled") {
                runtime = timeout_time();
            } else {
                tracing::error!(?reason, "Got unexpected unknown result");
                anyhow::bail!("{}", reason)
            }
        }

        drop(handle);

        Ok((
            Bench {
                id: SENTINEL_ID,
                problem_id: self.id,
                implementation,
                constraints,
                time_start,
                runtime,
                statistics: statistics.map(|s| s.into()),
                configuration: *configuration,
                iteration,
            },
            sat,
            !timed_out_thread,
        ))
    }
}
