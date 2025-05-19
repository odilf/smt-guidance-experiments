//! Export into CBOR format.

use std::{fs, io::Write, path::Path};

use minicbor::{Encoder, encode::write::Writer};
use rusqlite::{Connection, named_params};

/// Exports the bench data from the database in the CBOR format.
pub fn export(output_path: &Path, conn: &mut Connection) -> anyhow::Result<()> {
    let _span = tracing::info_span!("exporting", ?output_path).entered();

    let mut file = fs::File::options()
        .write(true)
        .create(true)
        .open(output_path)?;

    let mut encoder = Encoder::new(Writer::new(file.by_ref()));

    let mut stmt = conn.prepare(
        "SELECT problem.id, runtime, statistics FROM bench
            JOIN problem ON problem.id = bench.problem_id
        WHERE implementation = (:impl) AND tactic_help = (:tactic_help)",
    )?;

    let implementations = ["z3", "z3-noodler"];
    let tactic_helps = [0.0, 0.5];

    encoder.map(implementations.len() as u64)?;
    for implementation in implementations {
        encoder
            .str(implementation)?
            .map(tactic_helps.len() as u64)?;

        tracing::warn!("yo");
        for tactic_help in tactic_helps {
            let iter = stmt
                .query(named_params! {
                    ":impl": implementation,
                    ":tactic_help": tactic_help,
                })?
                .and_then(|row| anyhow::Ok((row.get(0)?, row.get(1)?, row.get(2)?)));

            for item in iter {
                let (problem_id, runtime, statistics): (u32, u64, String) = item?;
                encoder
                    .array(3)?
                    .u32(problem_id)?
                    .u64(runtime)?
                    // FIXME: This is JSON. Convert it back to CBOR.
                    .str(&statistics)?;
            }
        }
    }

    #[rustfmt::skip]
    encoder
        .map(2)?
        .str("z3")?.map(2)?
            .f32(0.0)?.begin_array()?;

    Ok(())
}
