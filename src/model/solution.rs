use std::borrow::Cow;

use anyhow::Context as _;
use rand::{Rng as _, SeedableRng as _, seq::IndexedRandom as _};
use serde::{Deserialize, Serialize};

use crate::model::bench::Bound;

use super::bench::Tactic;

#[derive(Debug, Clone)]
pub struct Solution {
    pub id: u32,
    pub sat: Sat,
    pub statistics: Statistics,
}

#[derive(Debug, Clone)]
pub enum Sat {
    /// The query is unsatisfiable.
    Unsat(UnsatCore),
    /// The query was interrupted, timed out or otherwise failed.
    Unknown,
    /// The query is satisfiable.
    Sat(Model),
}

impl Sat {
    pub fn to_str(&self) -> &'static str {
        match self {
            Self::Unsat(_) => "unsat",
            Self::Unknown => "unknown",
            Self::Sat(_) => "sat",
        }
    }

    pub fn model(&self) -> Option<&Model> {
        match self {
            Self::Sat(model) => Some(model),
            _ => None,
        }
    }
    pub fn unsat_core(&self) -> Option<&UnsatCore> {
        match self {
            Self::Unsat(unsat_core) => Some(unsat_core),
            _ => None,
        }
    }

    pub fn z3(&self) -> z3::SatResult {
        match self {
            Self::Sat(_) => z3::SatResult::Sat,
            Self::Unsat(_) => z3::SatResult::Unsat,
            Self::Unknown => z3::SatResult::Unknown,
        }
    }

    pub fn generate_tactic(&self, help: f32, problem_id: u32) -> Option<Tactic> {
        assert!((0.0..=1.0).contains(&help));

        // Deterministic rng.
        let mut rng = wyrand::WyRand::seed_from_u64(problem_id as u64);

        let bounds = match self {
            Sat::Unknown => return None,
            Sat::Sat(model) => {
                let mut r = help;
                // NOTE: Techinically, we could do statistical analysis to get the
                // expected value of the length of the vector and pre-allocate that,
                // but it seems unecessary...
                let mut output = Vec::new();
                while r > 0.0 {
                    r -= rng.random::<f32>();
                    let (name, value_repr) = model
                        .assignments
                        .choose(&mut rng)
                        .expect("Variable assignment should not be empty.");

                    let Some(value) = parse_value_repr(value_repr).unwrap() else {
                        tracing::debug!(
                            ?value_repr,
                            "skipping, since value repr doesn't seem to be a string"
                        );
                        continue;
                    };

                    let bound = Bound::new(&value, help, &mut rng);

                    // TODO: Maybe this doesn't need to be owned.
                    output.push((name.clone(), bound));
                }

                output
            }
            Sat::Unsat(_unsat_core) => {
                if help == 0.0 {
                    Vec::new()
                } else {
                    // TODO: Use tactics for `unsat_core`s
                    return None;
                }
            }
        };

        Some(Tactic { bounds, help })
    }
}

/// Parses the string representation of a z3 value into a Rust [`String`].
///
/// Returns [`Err`] if the parsing went wrong.
///
/// Returns [`None`] if the value is not a z3 string.
///
/// Panics if the input is malformed.
fn parse_value_repr(value_repr: &str) -> anyhow::Result<Option<Cow<'_, str>>> {
    // Strings are quoted.
    if !(value_repr.starts_with('"') && value_repr.starts_with('"')) {
        return Ok(None);
    }

    // Remove quotes.
    let value = &value_repr[1..value_repr.len() - 1];

    // Handle escaping
    if !value_repr.contains('\\') {
        return Ok(Some(Cow::Borrowed(value)));
    }

    let mut out = String::with_capacity(value_repr.len());
    let mut chars = value_repr
        .chars()
        .skip(1)
        .take(value_repr.chars().count() - 2)
        .peekable();

    while let Some(char) = chars.next() {
        if char != '\\' {
            out.push(char);
            continue;
        }

        type Iter<'a> = std::iter::Peekable<std::iter::Take<std::iter::Skip<std::str::Chars<'a>>>>;
        fn parse_escape(mut chars: Iter<'_>) -> anyhow::Result<(Vec<u8>, Iter<'_>)> {
            let next = chars.next().context("Backslash not followed by anything")?;
            if !next.eq_ignore_ascii_case(&'u') {
                anyhow::bail!("Backslash not followed by 'u' (is `{next}`)");
            }

            // From https://smt-lib.org/theories-UnicodeStrings.shtml, we can have the forms of escaping: ```
            // \ud₃d₂d₁d₀
            // \u{d₀}
            // \u{d₁d₀}
            // \u{d₂d₁d₀}
            // \u{d₃d₂d₁d₀}
            // \u{d₄d₃d₂d₁d₀}
            // ```

            let from_hex =
                |char: char| anyhow::Ok(char.to_digit(16).context("Not a hex digit")? as u8);
            let bytes = if *chars.peek().unwrap() == '{' {
                chars.next().unwrap();
                let mut bytes = Vec::new();
                loop {
                    let char = chars.next().context("No digits left")?;
                    if char == '}' {
                        break;
                    }

                    bytes.push(from_hex(char)?);
                }

                bytes
            } else {
                let d0 = from_hex(chars.next().context("No digits left")?)?;
                let d1 = from_hex(chars.next().context("No digits left")?)?;
                let d2 = from_hex(chars.next().context("No digits left")?)?;
                let d3 = from_hex(chars.next().context("No digits left")?)?;

                // TODO: Allocation is unfortunate...
                vec![d0, d1, d2, d3]
            };

            Ok((bytes, chars))
        }

        match parse_escape(chars.clone()) {
            Ok((bytes, advanced_chars)) => {
                let slice = String::from_utf8(bytes)?;
                out.push_str(&slice);
                chars = advanced_chars;
            }
            Err(err) => {
                tracing::warn!(
                    ?err,
                    "error while parsing escape value, interpreting it as-is."
                );
                out.push(char);
            }
        };
    }

    Ok(Some(Cow::Owned(out)))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    assignments: Vec<(String, String)>,
}

impl TryFrom<z3::Model<'_>> for Model {
    type Error = anyhow::Error;
    fn try_from(model: z3::Model<'_>) -> anyhow::Result<Self> {
        tracing::trace!(?model);
        Ok(Self {
            assignments: model
                .to_string()
                .lines()
                .map(|line| {
                    let (name, value) = line
                        .split_once(" -> ")
                        .context("No arrow in line of repr of model.")?;

                    Ok((name.to_string(), value.to_string()))
                })
                .collect::<anyhow::Result<_>>()?,
        })

        // NOTE: What I would like to do... But it just doesn't get the interpretations of the declarations :(
        // Ok(Self {
        //     assignments: model
        //         .iter()
        //         .map(|decl| {
        //             let name = decl.name(); // Name is simple

        //             // For value, we need to get the interpretation and the `get_else`?? For some reason?
        //             // I don't understand what it means, but whatever.
        //             let value = model
        //                 .get_func_interp(&decl)
        //                 .with_context(|| format!("No interpretation for {name} in model"))?
        //                 .get_else()
        //                 .as_string()
        //                 .context("Variable is not a string")?;

        //             Ok((name, value.to_string()))
        //         })
        //         .collect::<anyhow::Result<_>>()
        //         .unwrap(),
        // })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsatCore {
    // TODO: Give this more structure.
    reprs: Vec<String>,
}

impl From<Vec<z3::ast::Bool<'_>>> for UnsatCore {
    fn from(value: Vec<z3::ast::Bool<'_>>) -> Self {
        Self {
            reprs: value.into_iter().map(|bool| bool.to_string()).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    entries: Vec<(String, f64)>,
}

impl From<z3::Statistics<'_>> for Statistics {
    fn from(value: z3::Statistics<'_>) -> Self {
        Self {
            entries: value
                .entries()
                .map(|entry| {
                    let value = match entry.value {
                        z3::StatisticsValue::UInt(x) => f64::from(x),
                        z3::StatisticsValue::Double(x) => x,
                    };

                    (entry.key, value)
                })
                .collect(),
        }
    }
}
