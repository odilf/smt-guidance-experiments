use std::{
    borrow::Cow,
    fs,
    hash::{Hash, Hasher},
    path::Path,
    vec,
};

use jiff::Unit;
use rusqlite::{Connection, Row, named_params};
use serde_json::{from_str, to_string};
use walkdir::WalkDir;

use crate::model::{Bench, Problem, Solution, bench::Implementation, solution::Sat};

pub fn init(path: Option<impl AsRef<Path>>) -> anyhow::Result<Connection> {
    let conn = match path {
        Some(path) => Connection::open(path),
        None => Connection::open("./results.db"),
    }?;

    conn.execute_batch(
        "BEGIN;
        CREATE TABLE IF NOT EXISTS problem (
            id      INTEGER PRIMARY KEY,
            hash    INTEGER NOT NULL UNIQUE,
            path    TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS solution (
            id             INTEGER PRIMARY KEY,
            problem_id     INTEGER NOT NULL,
            implementation TEXT NOT NULL,
            sat            TEXT NOT NULL,
            model          TEXT NOT NULL,
            unsat_core     TEXT NOT NULL,
            statistics     TEXT NOT NULL,
            FOREIGN KEY(problem_id) REFERENCES problem(id)
        );
        CREATE TABLE IF NOT EXISTS bench (
            id             INTEGER PRIMARY KEY,
            problem_id     INTEGER NOT NULL,
            implementation TEXT NOT NULL,
            tactic_bounds  TEXT NOT NULL,
            tactic_help    FLOAT NOT NULL,
            time_start     TEXT NOT NULL,
            runtime        INTEGER NOT NULL,
            statistics     TEXT NOT NULL,
            configuration  TEXT NOT NULL,
            iteration      INTEGER NOT NULL,
            FOREIGN KEY(problem_id) REFERENCES problem(id)
        );
        CREATE INDEX IF NOT EXISTS idx_sat
            ON solution (sat);
        CREATE INDEX IF NOT EXISTS idx_implementation
            ON bench (implementation);
        CREATE INDEX IF NOT EXISTS idx_implementation_tactic_help
            ON bench (implementation, tactic_help);
        COMMIT;",
    )?;

    Ok(conn)
}

fn hash(program: &str) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    program.hash(&mut hasher);
    // NOTE: Divided by two because otherwise sqlite fails
    // with "out of range integral type conversion attempted"
    hasher.finish() / 2
}

pub fn populate(conn: &mut Connection, dataset_path: &Path, small: bool) -> anyhow::Result<()> {
    let _span = tracing::info_span!("populate db", ?dataset_path).entered();

    let mut stmt = conn.prepare(
        "INSERT OR IGNORE
        INTO problem (hash, path, content)
        VALUES(?1, ?2, ?3)",
    )?;
    let total = WalkDir::new(dataset_path).into_iter().count();

    tracing::info!("There are {total} entries");

    let step = if small { total / 1000 } else { 1 };
    for entry in WalkDir::new(dataset_path).into_iter().step_by(step) {
        let entry = entry?;
        if entry.path().is_dir() {
            tracing::debug!(dir=?entry.path(), "entering directory");
            continue;
        }

        tracing::trace!(file=?entry.path());
        let program = fs::read_to_string(entry.path())?;
        let hash = hash(&program);
        let rows_changed = stmt.execute((hash, entry.path().to_str(), program))?;
        match rows_changed {
            1 => tracing::trace!(hash, "Saved to db"),
            0 => tracing::debug!(hash, "Already in db"),
            _ => panic!(),
        }
    }

    Ok(())
}

pub fn iter_unsolved_problems(implementation: Implementation) -> PagedIterator<Problem> {
    // XXX
    // Cow::Borrowed(
    //     "SELECT problem.id, hash, path, content FROM problem
    //     WHERE solution.problem_id IS NULL
    //         AND solution.implementation
    //     LEFT JOIN solution ON problem.id = solution.problem_id
    //     LIMIT (:limit) OFFSET (:offset)",
    // ),
    let query = format!(
        r#"SELECT problem.id, hash, path, content FROM problem
	    WHERE ("{implementation}") not in (
	        SELECT implementation FROM solution
	        WHERE solution.problem_id = problem.id
	    )
	    LIMIT (:limit) OFFSET (:offset)"#
    );

    PagedIterator::new(Cow::Owned(query), |row| {
        Ok(Problem {
            id: row.get(0)?,
            hash: row.get(1).unwrap_or(69),
            path: row.get::<_, String>(2)?.into(),
            content: row.get(3)?,
        })
    })
}

pub fn iter_unbenched_problems(
    implementation: Implementation,
    tactic_help: f32,
    iteration: u16,
) -> PagedIterator<(Problem, Solution)> {
    let query = format!(
        r#"SELECT problem.id AS problem_id, hash, path, content,
            solution.id AS solution_id, solution.implementation AS solution_impl, sat, model, unsat_core, solution.statistics AS sol_statistics
            FROM problem
            JOIN solution ON problem.id = solution.problem_id AND solution.implementation = "{implementation}"
	    WHERE ("{implementation}", {tactic_help}) not in (
	        SELECT bench.implementation, bench.tactic_help FROM bench
	        WHERE bench.problem_id = problem.id AND bench.iteration = {iteration}
	    )
	    LIMIT (:limit) OFFSET (:offset)"#
    );

    tracing::debug!(?query);

    PagedIterator::new(Cow::Owned(query), |row| {
        let problem = Problem {
            id: row.get("problem_id")?,
            hash: row.get("hash")?,
            path: row.get::<_, String>("path")?.into(),
            content: row.get("content")?,
        };

        let sat = match row.get::<_, String>("sat")?.as_ref() {
            "sat" => Sat::Sat(from_str(row.get::<_, String>("model")?.as_ref())?),
            "unsat" => Sat::Unsat(from_str(row.get::<_, String>("unsat_core")?.as_ref())?),
            "unknown" => Sat::Unknown,
            x => unreachable!("Value is `{x}`"),
        };

        let solution = Solution {
            id: row.get("solution_id")?,
            implementation: Implementation::from_str(
                row.get::<_, String>("solution_impl")?.as_ref(),
            )?,
            sat,
            statistics: from_str(row.get::<_, String>("sol_statistics")?.as_ref())?,
        };

        Ok((problem, solution))
    })
}

pub struct PagedIterator<T> {
    query: Cow<'static, str>,
    into: fn(&Row<'_>) -> anyhow::Result<T>,
    page_index: usize,
    page_iter: vec::IntoIter<T>,
}

impl<T> PagedIterator<T> {
    const PAGE_SIZE: usize = 100;

    pub fn new(query: Cow<'static, str>, into: fn(&Row<'_>) -> anyhow::Result<T>) -> Self {
        Self {
            query,
            into,
            page_index: 0,
            page_iter: vec![].into_iter(),
        }
    }

    fn get_page(&self, conn: &mut Connection) -> anyhow::Result<Vec<T>> {
        let mut stmt = conn.prepare(self.query.as_ref())?;

        stmt.query(named_params! {
            ":limit": Self::PAGE_SIZE,
            ":offset": Self::PAGE_SIZE * self.page_index,
        })?
        .and_then(|row| (self.into)(row))
        .collect()
    }

    fn advance_page(&mut self, conn: &mut Connection) -> anyhow::Result<()> {
        tracing::trace!(?self.page_index);
        self.page_iter = self.get_page(conn)?.into_iter();
        self.page_index += 1;

        Ok(())
    }

    pub fn next(&mut self, conn: &mut Connection) -> anyhow::Result<Option<T>> {
        if let Some(elem) = self.page_iter.next() {
            return Ok(Some(elem));
        }

        self.advance_page(conn)?;
        Ok(self.page_iter.next())
    }
}

pub fn get_number_of_problems(conn: &mut Connection) -> rusqlite::Result<usize> {
    // TODO: This takes waaaaaaaaaaaaaaay too long!
    conn.query_row("SELECT COUNT(1) FROM problem", (), |row| row.get(0))
}

pub fn insert_solution(
    conn: &mut Connection,
    solution: &Solution,
    problem: &Problem,
) -> anyhow::Result<()> {
    let rows_changed = conn.execute(
        "INSERT INTO solution (problem_id, implementation, sat, model, unsat_core, statistics) VALUES ($1, $2, $3, $4, $5, $6)",
        (
            problem.id,
            &solution.implementation.to_str(),
            &solution.sat.to_str(),
            &to_string(&solution.sat.model()).unwrap(),
            &to_string(&solution.sat.unsat_core()).unwrap(),
            &to_string(&solution.statistics).unwrap(),
        ),
    )?;

    assert_eq!(rows_changed, 1);
    tracing::debug!("Saved to db");

    Ok(())
}

pub fn insert_bench(conn: &mut Connection, bench: &Bench, problem: &Problem) -> anyhow::Result<()> {
    let runtime_micros = bench.runtime.total(Unit::Microsecond).unwrap();

    let rows_changed = conn.execute(
        "INSERT INTO bench (
            problem_id,
            implementation,
            tactic_bounds,
            tactic_help,
            time_start,
            runtime,
            statistics,
            configuration,
            iteration
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        (
            problem.id,
            &bench.implementation.to_str(),
            &to_string(&bench.tactic.bounds).unwrap(),
            &to_string(&bench.tactic.help).unwrap(),
            &to_string(&bench.time_start).unwrap(),
            &runtime_micros,
            &to_string(&bench.statistics).unwrap(),
            &to_string(&bench.configuration).unwrap(),
            &bench.iteration,
        ),
    )?;

    assert_eq!(rows_changed, 1);
    tracing::debug!("Saved to db");

    Ok(())
}
