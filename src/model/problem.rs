use std::path::PathBuf;

use anyhow::Context as _;

use super::{
    Bench, Solution,
    bench::{Config, Implementation, Tactic},
    solution::Sat,
};

const SENTINEL_ID: u32 = 69420;

// TODO: Would be nice to extract the `set-info` parts of the original files.
#[derive(Debug, Clone)]
pub struct Problem {
    pub id: u32,
    pub path: PathBuf,
    pub content: String,
}

impl Problem {
    pub fn solve(&self, solver: &z3::Solver) -> anyhow::Result<Solution> {
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
            sat,
            statistics,
        })
    }

    pub fn bench_with_tactic(
        &self,
        solver: &z3::Solver<'_>,
        tactic: Tactic,
        implementation: Implementation,
        configuration: &Config,
        iteration: u16,
    ) -> anyhow::Result<(Bench, z3::SatResult)> {
        solver.from_string(&*self.content);

        tactic.add_bounds_to_solver(solver);

        let time_start = jiff::Timestamp::now();
        let sat = solver.check();
        let time_end = jiff::Timestamp::now();

        let runtime = time_end - time_start;

        tracing::debug!(?runtime);

        if solver
            .get_reason_unknown()
            .is_some_and(|reason| reason.contains("interrupted"))
        {
            anyhow::bail!("Interrupted.");
        }

        Ok((
            Bench {
                id: SENTINEL_ID,
                problem_id: self.id,
                implementation,
                tactic,
                time_start,
                runtime,
                statistics: solver.get_statistics().into(),
                configuration: *configuration,
                iteration,
            },
            sat,
        ))
    }
}
