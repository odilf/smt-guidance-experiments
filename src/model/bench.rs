use std::{borrow::Cow, fmt, time::Duration};

use anyhow::Context as _;
use clap::ValueEnum;
use rand::{seq::IndexedRandom as _, Rng};
use rand_distr::uniform::SampleRange;
use serde::{Deserialize, Serialize};

use super::solution::Statistics;

/// A single result from a benchmark.
#[derive(Debug, Clone)]
pub struct Bench {
    pub id: u32,
    pub problem_id: u32,
    pub implementation: Implementation,
    /// [`Constraints`] used to solve this instance. If no constraints were used, it is represented by having `help` field be `0.0`.
    pub constraints: Constraints,
    pub time_start: jiff::Timestamp,
    pub runtime: jiff::Span,
    pub statistics: Option<Statistics>,
    /// Configuration of the solver used.
    pub configuration: Config,
    pub iteration: u16,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Implementation {
    Z3str3,
    Z3Noodler,
}

impl Implementation {
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Z3str3 => "z3str3",
            Self::Z3Noodler => "z3-noodler",
        }
    }

    pub fn from_str(input: &str) -> anyhow::Result<Implementation> {
        let implementation = match input {
            "z3str3" => Self::Z3str3,
            "z3-noodler" => Self::Z3Noodler,
            other => anyhow::bail!("Uknown z3 implementation: {other}"),
        };

        Ok(implementation)
    }

    pub fn from_env() -> anyhow::Result<Self> {
        Self::from_str(
            // Random characters to discourage a user from setting it.
            // It should be set in the Nix environment.
            std::env::var("BEFDLSKSEF_EXPERIMENTS_Z3_IMPLEMENTATION")
                .context("Environment has not set a z3 implementation.")?
                .as_str(),
        )
    }
}

impl fmt::Display for Implementation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.to_str())
    }
}

impl ValueEnum for Implementation {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Z3str3, Self::Z3Noodler]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(self.to_str().into())
    }
}

/// Tailored constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraints {
    /// The added bounds for each variable, as a `(var_name, value, bound)` tuple.
    pub bounds: Vec<(String, String, Constraint)>,

    /// How much help this gives, from 0.0 to 1.0.
    pub help: f32,
}

impl Constraints {
    pub fn add_bounds_to_solver(&self, solver: &z3::Solver<'_>) {
        for (var_name, value, bound) in &self.bounds {
            solver.from_string(bound.to_string(var_name, value));
        }
    }
}

/// Possible bounds for [`Constraints`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Constraint {
    LengthGE { length: u32 },
    LengthLT { length: u32 },
    LengthEquals,
    PrefixOf { length: u32 },
    SuffixOf { length: u32 },
    Contains { start: u32, length: u32 },
}

impl Constraint {
    /// The amount of help this constraint gives given the distribution parameters C and lambda.
    pub fn actual_help(self, value: &str, c: f32, lambda: f32) -> f32 {
        let len = value.chars().count() as f32;
        let (length_guess_prob_ln, _, guess_prob_ln) =
            Self::raw_guess_probabilities_ln(value, c, lambda);

        let increase_ln = match self {
            Self::LengthGE { length } => lambda * length as f32,
            Constraint::LengthLT { length } => -(1.0 - (-lambda * length as f32).exp()).ln(),
            Constraint::LengthEquals => length_guess_prob_ln,
            Constraint::PrefixOf { length } => (length as f32) * (c.ln() + lambda),
            Constraint::SuffixOf { length } => (length as f32) * (c.ln() + lambda),
            Constraint::Contains { length, .. } => {
                ((length as f32) * (c.ln() + lambda) - (len - length as f32 + 1.0).ln()).max(0.0)
            }
        };

        let help = -increase_ln / guess_prob_ln;

        assert!((0.0..=1.0).contains(&help), "Help was {help}");
        help
    }

    /// Returns `(length_guess_prob_ln, char_guess_prob_ln, guess_prob_ln)`.
    fn raw_guess_probabilities_ln(value: &str, c: f32, lambda: f32) -> (f32, f32, f32) {
        let len = value.chars().count() as f32;

        let length_guess_prob_ln = -lambda * len + (1.0 - (-lambda).exp()).ln();
        let char_guess_prob_ln = -len * c.ln();
        let guess_prob_ln = length_guess_prob_ln + char_guess_prob_ln;

        (length_guess_prob_ln, char_guess_prob_ln, guess_prob_ln)
    }

    fn length_ge(value: &str, help: f32, c: f32, lambda: f32, _: &mut impl Rng) -> Self {
        let (_, _, guess_prob_ln) = Self::raw_guess_probabilities_ln(value, c, lambda);
        let ell = (help * guess_prob_ln) / lambda;
        let length = ell.floor().min(value.chars().count() as f32) as u32;
        Self::LengthGE { length }
    }

    fn length_lt(value: &str, help: f32, c: f32, lambda: f32, _: &mut impl Rng) -> Self {
        let (_, _, guess_prob_ln) = Self::raw_guess_probabilities_ln(value, c, lambda);
        // TODO: The `guess_prob_ln.exp()` is kind of wasteful.
        let ell = (1.0 - help.exp() * guess_prob_ln.exp()).ln() / -lambda;
        let ell = ell.max(value.chars().count() as f32 + 1.0);
        let length = ell.floor() as u32;
        Self::LengthLT { length }
    }

    fn length_eq(_value: &str, _help: f32, _c: f32, _lambda: f32, _: &mut impl Rng) -> Self {
        Self::LengthEquals
    }

    fn prefix_of(value: &str, help: f32, c: f32, lambda: f32, _: &mut impl Rng) -> Self {
        let (_, _, guess_prob_ln) = Self::raw_guess_probabilities_ln(value, c, lambda);
        let ell = help * -guess_prob_ln / (c.ln() + lambda);
        let length = ell.floor().min(value.chars().count() as f32) as u32;
        Self::PrefixOf { length }
    }

    fn suffix_of(value: &str, help: f32, c: f32, lambda: f32, _: &mut impl Rng) -> Self {
        let (_, _, guess_prob_ln) = Self::raw_guess_probabilities_ln(value, c, lambda);
        let ell = help * -guess_prob_ln / (c.ln() + lambda);
        let length = ell.floor().min(value.chars().count() as f32) as u32;
        Self::SuffixOf { length }
    }

    fn contains(value: &str, help: f32, c: f32, lambda: f32, rng: &mut impl Rng) -> Self {
        let (_, _, guess_prob_ln) = Self::raw_guess_probabilities_ln(value, c, lambda);
        let len = value.chars().count() as u32;
        let help_of = |partial_len: f32| {
            ((partial_len * c.ln() - (len as f32 - partial_len + 1.0).ln()) / -guess_prob_ln)
                .max(0.0)
        };

        // Find highest lower bound of length with binary search.
        let mut lower = 0;
        let mut upper = len;
        let length = loop {
            if upper - lower <= 1 {
                break lower;
            }

            let guess_length = (upper + lower) / 2;
            let guess_help = help_of(guess_length as f32);
            if guess_help > help {
                upper = guess_length;
            } else if guess_help < help {
                lower = guess_length;
            } else {
                break guess_length;
            }
        };

        let start = rng.random_range(0..(len - length + 1));

        Constraint::Contains { start, length }
    }

    /// Returns (C, lambda).
    pub fn get_distribution_parameters(value: &str) -> (f32, f32) {
        const ACTUAL_C: f32 = 3.0 * 16.0 * 16.0 * 16.0 * 16.0;
        let c = ACTUAL_C;
        // let c = value.chars().collect::<HashSet<_>>().len() as f32;
        // let lambda = 1.0 / (value.chars().count() as f32);
        let len = value.chars().count() as f32;
        let lambda = if len == 0.0 {
            1.0
        } else {
            ((len + 1.0) / len).ln()
        };

        (c, lambda)
    }

    /// Creates a new random bound and gives the help it was actually assigned, which
    /// is the closest possible value to a lower bound of the target `help`.
    pub fn new_single(value: &str, help: f32, rng: &mut impl Rng) -> Self {
        let constructor = [
            Self::length_ge,
            Self::length_lt,
            Self::length_eq,
            Self::prefix_of,
            Self::suffix_of,
            Self::contains,
        ]
        .choose(rng)
        .unwrap();

        let (c, lambda) = Self::get_distribution_parameters(value);

        constructor(value, help, c, lambda, rng)
    }

    /// Returns an iterator of bounds that together result in the closest
    /// possible value to the given help.
    pub fn new<R: Rng>(
        value: &str,
        mut help: f32,
        rng: &mut R,
    ) -> impl Iterator<Item = Constraint> + use<R> {
        let mut output = Vec::new();

        let mut constructors = vec![
            Self::length_ge,
            Self::length_lt,
            Self::prefix_of,
            Self::suffix_of,
            Self::contains,
        ];

        let (c, lambda) = Self::get_distribution_parameters(value);

        while help > 0.01 {
            let Ok(i) = (0..constructors.len()).sample_single(rng) else {
                break;
            };

            // We can swap remove because each item has the same probability, so order doesn't matter.
            let constructor = constructors.swap_remove(i);
            let bound = constructor(value, help, c, lambda, rng);
            let actual_help = bound.actual_help(value, c, lambda);

            // TODO: This infinite loops
            if actual_help > help {
                break;
            }

            if actual_help == 0.0 {
                // Don't add useless bounds.
                continue;
            }

            help -= actual_help;
            output.push(bound);
        }

        output.into_iter()
    }

    // TODO: This has way too many allocations. Really, shouldn't return a
    // String, just impl Write or something
    pub fn to_string(self, var_name: &str, value: &str) -> String {
        let var = escape_for_z3(var_name);
        match self {
            Self::PrefixOf { length } => format!(
                "(assert (str.prefixof \"{prefix}\" {var}))",
                prefix = escape_for_z3(&value.chars().take(length as usize).collect::<String>()),
            ),
            Self::SuffixOf { length } => format!(
                "(assert (str.suffixof \"{suffix}\" {var}))",
                suffix = escape_for_z3(
                    &value
                        .chars()
                        .skip(value.chars().count() - length as usize)
                        .collect::<String>(),
                )
            ),
            Self::Contains { start, length } => format!(
                "(assert (str.contains {var} \"{substring}\"))",
                substring = escape_for_z3(
                    &value
                        .chars()
                        .skip(start as usize)
                        .take(length as usize)
                        .collect::<String>()
                )
            ),
            // Lol, reverse order
            Self::LengthEquals => format!(
                "(assert (== (str.len {var}) {len}))",
                len = value.chars().count()
            ),
            Self::LengthGE { length } => format!("(assert (>= (str.len {var}) {length}))"),
            Self::LengthLT { length } => format!("(assert (< (str.len {var}) {length}))"),
        }
    }

    /// User-facing method
    pub fn help_given_solution(&self, solution: &str) -> f32 {
        let (c, lambda) = Self::get_distribution_parameters(solution);
        self.actual_help(solution, c, lambda)
    }
}

/// Escapes strings with null bytes using SMTLIB2 standard (i.e., `\u{0}`).
///
/// Also escape quotes, to be able to interpolate them.
///
/// This is needed because the z3 Rust bindings convert the string into a [`CString`](std::ffi::CString).
pub fn escape_for_z3(input: &str) -> Cow<'_, str> {
    let mut output = Cow::Borrowed(input);
    // if output.contains('\0') {
    //     output = Cow::Owned(output.replace('\0', r#"\u{0}"#))
    // }

    if output.chars().any(|c| !c.is_ascii() || c.is_control()) {
        let mut escaped = String::new();
        for char in output.chars() {
            if !char.is_ascii() || char.is_control() {
                escaped.extend(char.escape_unicode())
            } else {
                escaped.push(char)
            }
        }
        output = Cow::Owned(escaped)
    }

    if output.contains('"') {
        output = Cow::Owned(output.replace('"', "\"\""));
    }

    output
}

impl From<&str> for Constraint {
    fn from(value: &str) -> Self {
        if let Some(arg) = value.strip_prefix("length-ge") {
            let length = arg.parse().unwrap();
            Self::LengthGE { length }
        } else if let Some(arg) = value.strip_prefix("length-lt") {
            let length = arg.parse().unwrap();
            Self::LengthLT { length }
        } else if value == "length-eq" {
            Self::LengthEquals
        } else if let Some(arg) = value.strip_prefix("prefix") {
            let length = arg.parse().unwrap();
            Self::PrefixOf { length }
        } else if let Some(arg) = value.strip_prefix("suffix") {
            let length = arg.parse().unwrap();
            Self::SuffixOf { length }
        } else if let Some(arg) = value.strip_prefix("contains") {
            let length = arg.parse().unwrap();
            // Start doesn't affect help calculation.
            Self::Contains { length, start: 0 }
        } else {
            todo!("Implement ")
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Config {
    proof_generation: bool,
    model_generation: bool,
    pub timeout: Duration,
    debug_ref_count: bool,
    // TODO: These don't seem to work, for some reason... :(
    memory_high_watermark_mb: usize,
    memory_max_size_mb: usize,
}

impl Config {
    pub fn gen_model() -> Self {
        Self {
            // proof_generation: true,
            proof_generation: false,
            model_generation: true,
            timeout: Duration::from_secs(370),
            debug_ref_count: false,
            memory_high_watermark_mb: 3500,
            memory_max_size_mb: 5000,
        }
    }

    pub fn benchmark() -> Self {
        Self {
            proof_generation: false,
            model_generation: false,
            // timeout: Duration::from_secs(60 * 16),
            timeout: Duration::from_secs(360),
            debug_ref_count: false,
            memory_high_watermark_mb: 4000,
            memory_max_size_mb: 6000,
        }
    }

    pub fn z3(&self) -> z3::Config {
        let mut cfg = z3::Config::new();
        cfg.set_proof_generation(self.proof_generation);
        cfg.set_model_generation(self.model_generation);
        cfg.set_timeout_msec(u64::try_from(self.timeout.as_millis()).unwrap());
        cfg.set_debug_ref_count(self.debug_ref_count);
        // Apparently these don't work?
        // cfg.set_param_value("memory_high_watermark_mb", &self.memory_high_watermark_mb);
        // cfg.set_param_value("memory_max_size", self.memory_max_size_mb);
        cfg
    }
}
