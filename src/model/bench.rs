use std::{borrow::Cow, fmt, time::Duration};

use clap::ValueEnum;
use rand::{Rng, seq::IndexedRandom as _};
use serde::{Deserialize, Serialize};

use super::solution::Statistics;

/// A single result from a benchmark.
#[derive(Debug, Clone)]
pub struct Bench {
    pub id: u32,
    pub problem_id: u32,
    pub implementation: Implementation,
    /// Tactic used to solve this instance. If no tactic was used, it is represented by having `help` field be `0.0`.
    pub tactic: Tactic,
    pub time_start: jiff::Timestamp,
    pub runtime: jiff::Span,
    pub statistics: Statistics,
    /// Configuration of the solver used.
    pub configuration: Config,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Implementation {
    Z3,
    Z3Noodler,
    Z3Trau,
}

impl Implementation {
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Z3 => "z3",
            Self::Z3Noodler => "z3-noodler",
            Self::Z3Trau => "z3-trau",
        }
    }
}

impl fmt::Display for Implementation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.to_str())
    }
}

impl ValueEnum for Implementation {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Z3, Self::Z3Noodler, Self::Z3Trau]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(self.to_str().into())
    }
}

/// A tailored tactic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tactic {
    /// The added bounds of this tactic for each variable.
    pub bounds: Vec<(String, Bound)>,

    /// How much help this tactic gives, from 0.0 to 1.0.
    pub help: f32,
}

impl Tactic {
    pub fn add_bounds_to_solver(&self, solver: &z3::Solver<'_>) {
        for (variable, bound) in &self.bounds {
            solver.from_string(bound.to_string(variable));
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Possible bounds for [`Tactic`]s.
pub enum Bound {
    StartsWith(String),
    EndsWith(String),
    Contains(String),
    LengthEquals(u32),
    LengthGE(u32),
    LengthLE(u32),
    Or(Vec<Bound>),
    And(Vec<Bound>),
    // TODO: Check if there are more
}

impl Bound {
    fn starts_with(value: &str, help: f32, _: &mut impl Rng) -> Self {
        let len = (value.chars().count() as f32 * help) as usize;
        Self::StartsWith(value.chars().take(len).collect())
    }

    fn ends_with(value: &str, help: f32, _: &mut impl Rng) -> Self {
        let vlen = value.chars().count();
        let len = (vlen as f32 * help) as usize;
        Self::EndsWith(value.chars().skip(vlen - len).collect())
    }

    fn contains(value: &str, help: f32, rng: &mut impl Rng) -> Self {
        let vlen = value.chars().count();
        let start_percentage = rng.random::<f32>();
        let start = (vlen as f32 * start_percentage) as usize;
        let len = (vlen as f32 * help) as usize;

        Self::Contains(value.chars().skip(start).take(len).collect())
    }

    fn length_ge(value: &str, help: f32, _: &mut impl Rng) -> Self {
        let vlen = value.chars().count();
        // TODO: Change to make impact the same as starts_with, ends_with
        Self::LengthGE((vlen as f32 * help) as u32)
    }

    fn length_le(value: &str, help: f32, _: &mut impl Rng) -> Self {
        let vlen = value.chars().count();
        // TODO: Change to make impact the same as starts_with, ends_with
        Self::LengthLE((vlen as f32 / help).ceil() as u32)
    }

    pub fn new(value: &str, help: f32, rng: &mut impl Rng) -> Self {
        let constructor = [
            Self::starts_with,
            Self::ends_with,
            Self::contains,
            // Self::length_ge,
            // Self::length_le,
        ]
        .choose(rng)
        .unwrap();

        constructor(value, help, rng)
    }

    fn to_string(&self, var: &str) -> String {
        let var = escape_null_byte(var);
        match self {
            Self::StartsWith(prefix) => format!(
                "(assert (str.prefixof {prefix} {var})",
                prefix = escape_null_byte(prefix)
            ),
            Self::EndsWith(suffix) => format!(
                "(assert (str.suffixof {suffix} {var})",
                suffix = escape_null_byte(suffix)
            ),
            Self::Contains(substring) => format!(
                "(assert (str.contains {var} {substring})",
                substring = escape_null_byte(substring)
            ), // Lol, reverse order
            Self::LengthEquals(len) => format!("(assert (== (str.len {var}) {len}))"),
            Self::LengthGE(len) => format!("(assert (>= (str.len {var}) {len}))"),
            Self::LengthLE(len) => format!("(assert (<= (str.len {var}) {len}))"),
            Self::Or(_) => todo!(),
            Self::And(_) => todo!(),
        }
    }
}

/// Escapes strings with null bytes using SMTLIB2 standard (i.e., `\u{0}`).
///
/// This is needed because the z3 Rust bindings convert the string into a [`CString`](std::ffi::CString).
pub fn escape_null_byte(input: &str) -> Cow<'_, str> {
    if !input.contains('\0') {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(input.replace('\0', r#"\u{0}"#))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Config {
    proof_generation: bool,
    model_generation: bool,
    timeout: Duration,
    debug_ref_count: bool,
}

impl Config {
    pub fn gen_model() -> Self {
        Self {
            proof_generation: true,
            model_generation: true,
            timeout: Duration::from_secs(60 * 5),
            debug_ref_count: false,
        }
    }

    pub fn benchmark() -> Self {
        Self {
            proof_generation: false,
            model_generation: false,
            timeout: Duration::from_secs(60 * 5),
            debug_ref_count: false,
        }
    }

    pub fn z3<'a>(&self) -> z3::Config {
        let mut cfg = z3::Config::new();
        cfg.set_proof_generation(self.proof_generation);
        cfg.set_model_generation(self.model_generation);
        cfg.set_timeout_msec(u64::try_from(self.timeout.as_millis()).unwrap());
        cfg.set_debug_ref_count(self.debug_ref_count);
        cfg
    }
}
