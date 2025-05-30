use std::{borrow::Cow, fmt};

use anyhow::Context;
use rand::SeedableRng as _;
use serde::{Deserialize, Serialize};

use crate::model::bench::Constraint;

use super::bench::{Constraints, Implementation};

#[derive(Debug, Clone)]
pub struct Solution {
    pub id: u32,
    pub implementation: Implementation,
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
    #[allow(clippy::enum_variant_names)]
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

    /// # Panics
    ///
    /// If `help` is NaN.
    pub fn generate_constraints(&self, help: f32, problem_hash: u64) -> Option<Constraints> {
        assert!((0.0..=1.0).contains(&help));

        // Deterministic rng.
        let mut rng = wyrand::WyRand::seed_from_u64(problem_hash);

        let Sat::Sat(model) = self else {
            return None;
        };

        let bounds = {
            let variables = &model.assignments;

            let helps = split_ranges(variables.len(), &mut rng)
                .into_iter()
                .map(|r| r * help);

            helps
                .zip(variables)
                .flat_map(|(help, (name, value))| {
                    Constraint::new(value, help, &mut rng)
                        .map(|bound| (name.to_string(), value.clone(), bound))
                })
                .collect()
        };

        Some(Constraints { bounds, help })
    }

    pub const fn is_sat(&self) -> bool {
        matches!(self, Self::Sat(_))
    }
}

/// Splits the range `[0, max)` into `n` segments using the given `rng`.
///
/// The returns is a vec of numbers that add up to `help`.
// TODO: Unit test function.
fn split_ranges(n: usize, rng: &mut impl rand::Rng) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }

    let mut values = vec![0.0f32; n];

    // Fill with random values and sort, except first to keep a 0 at the front.
    rng.fill(&mut values[1..]);
    values[1..].sort_by(|a, b| a.total_cmp(b));

    // Calculate range sizes.
    for i in 0..values.len() {
        values[i] = values.get(i + 1).unwrap_or(&1.0) - values[i];
    }

    values
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

    let mut output = Cow::Borrowed(value);

    // // Handle escaping
    // if !value_repr.contains('\\') {
    //     return Ok(Some(Cow::Borrowed(value)));
    // }

    // return Ok(Some(Cow::Owned(claude_parse_unicode_escapes_manual(
    //     value,
    // )?)));

    output = Cow::Owned(output.replace("\"\"", "\""));
    output = Cow::Owned(claude_parse_unicode_escapes_manual(&output)?);

    Ok(Some(output))

    // let mut out = String::with_capacity(value_repr.len());
    // let mut chars = value_repr
    //     .chars()
    //     .skip(1)
    //     .take(value_repr.chars().count() - 2)
    //     .peekable();

    // while let Some(char) = chars.next() {
    //     if char == '"' {
    //         out.push('"');

    //         // Consume two quotes as one, if there are. Otherwise, continue as normal.
    //         if chars.peek() == Some(&'"') {
    //             chars.next();
    //         }

    //         continue;
    //     } else if char != '\\' {
    //         out.push(char);
    //         continue;
    //     }

    //     type Iter<'a> = std::iter::Peekable<std::iter::Take<std::iter::Skip<std::str::Chars<'a>>>>;
    //     fn parse_escape(mut chars: Iter<'_>) -> anyhow::Result<(Vec<u8>, Iter<'_>)> {
    //         let next = chars.next().context("Backslash not followed by anything")?;
    //         if !next.eq_ignore_ascii_case(&'u') {
    //             anyhow::bail!("Backslash not followed by 'u' (is `{next}`)");
    //         }

    //         // From https://smt-lib.org/theories-UnicodeStrings.shtml, we can have the forms of escaping: ```
    //         // \ud₃d₂d₁d₀
    //         // \u{d₀}
    //         // \u{d₁d₀}
    //         // \u{d₂d₁d₀}
    //         // \u{d₃d₂d₁d₀}
    //         // \u{d₄d₃d₂d₁d₀}
    //         // ```

    //         let from_hex =
    //             |char: char| anyhow::Ok(char.to_digit(16).context("Not a hex digit")? as u8);
    //         let bytes = if *chars.peek().unwrap() == '{' {
    //             chars.next().unwrap();
    //             let mut bytes = Vec::new();
    //             loop {
    //                 let char = chars.next().context("No digits left")?;
    //                 if char == '}' {
    //                     break;
    //                 }

    //                 bytes.push(from_hex(char)?);
    //             }

    //             bytes
    //         } else {
    //             let d0 = from_hex(chars.next().context("No digits left")?)?;
    //             let d1 = from_hex(chars.next().context("No digits left")?)?;
    //             let d2 = from_hex(chars.next().context("No digits left")?)?;
    //             let d3 = from_hex(chars.next().context("No digits left")?)?;

    //             // TODO: Allocation is unfortunate...
    //             vec![d0 + d1 * 16, d2 + d3 * 16]
    //         };

    //         Ok((bytes, chars))
    //     }

    //     match parse_escape(chars.clone()) {
    //         Ok((bytes, advanced_chars)) => {
    //             let slice = String::from_utf8(bytes)?;
    //             out.push_str(&slice);
    //             chars = advanced_chars;
    //         }
    //         Err(err) => {
    //             tracing::warn!(
    //                 ?err,
    //                 "error while parsing escape value, interpreting it as-is."
    //             );
    //             out.push(char);
    //         }
    //     };
    // }

    // Ok(Some(Cow::Owned(out)))
}

pub fn claude_parse_unicode_escapes_manual(input: &str) -> anyhow::Result<String> {
    let mut result = String::new();
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' && chars.peek() == Some(&'u') {
            chars.next(); // consume 'u'

            let code_point = if chars.peek() == Some(&'{') {
                // Braced format: \u{...}
                chars.next(); // consume '{'

                let mut hex_digits = String::new();
                while let Some(&digit) = chars.peek() {
                    if digit == '}' {
                        chars.next(); // consume '}'
                        break;
                    }
                    if digit.is_ascii_hexdigit() {
                        hex_digits.push(chars.next().unwrap());
                    } else {
                        anyhow::bail!("Invalid hex digit.");
                    }
                }

                // Validate length (1-5 digits)
                if hex_digits.is_empty() || hex_digits.len() > 5 {
                    anyhow::bail!("Length is invalid: {}", hex_digits.len());
                }

                // Validate d₄ restriction for 5-digit sequences
                if hex_digits.len() == 5 {
                    let first_digit = hex_digits.chars().next().unwrap();
                    if !matches!(first_digit, '0'..='2') {
                        anyhow::bail!("Invalid hex digit.");
                    }
                }

                u32::from_str_radix(&hex_digits, 16).context("Invalid hex digit.")?
            } else {
                // Fixed format: \udddd (4 digits)
                let mut hex_digits = String::new();
                for _ in 0..4 {
                    if let Some(digit) = chars.next() {
                        if digit.is_ascii_hexdigit() {
                            hex_digits.push(digit);
                        } else {
                            anyhow::bail!("Invalid hex digit.");
                        }
                    } else {
                        anyhow::bail!("Invalid hex digit.");
                    }
                }

                u32::from_str_radix(&hex_digits, 16).context("Invalid hex digit.")?
            };

            let character = char::from_u32(code_point).context("Invalid hex digit.")?;

            result.push(character);
        } else {
            result.push(ch);
        }
    }

    Ok(result)
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
                .filter_map(|line| {
                    let (name, value) = line.split_once(" -> ")?;
                    let value = parse_value_repr(value).unwrap()?;

                    Some((name.to_string(), value.to_string()))
                })
                .collect(),
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

impl fmt::Display for Sat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.to_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        const N: usize = 100_000;
        let size = 10;
        let mut bins = vec![Vec::with_capacity(N); size];
        for _ in 0..N {
            let output = split_ranges(bins.len(), &mut rand::rng());
            assert!(
                (output.iter().sum::<f32>() - 1.0).abs() < 1e-8,
                "output was {output:?}, sum is {}",
                output.iter().sum::<f32>()
            );

            for (bin, output) in bins.iter_mut().zip(output) {
                bin.push(output);
            }
        }

        for bin in bins {
            assert!((bin.iter().sum::<f32>() / bin.len() as f32 - 1.0 / size as f32) < 1e-2);
        }
    }

    #[test]
    fn test_fixed_format() {
        assert_eq!(claude_parse_unicode_escapes_manual("\\u0041").unwrap(), "A");
        assert_eq!(
            claude_parse_unicode_escapes_manual("Hello \\u0041 World").unwrap(),
            "Hello A World"
        );
    }

    #[test]
    fn test_braced_format() -> anyhow::Result<()> {
        assert_eq!(claude_parse_unicode_escapes_manual("\\u{13}")?, "\u{13}");
        assert_eq!(claude_parse_unicode_escapes_manual("\\u{41}")?, "A");
        assert_eq!(claude_parse_unicode_escapes_manual("\\u{1F600}")?, "😀");
        // assert_eq!(
        //     claude_parse_unicode_escapes_manual("\\u{10FFFF}")?,
        //     "\u{10FFFF}"
        // );

        Ok(())
    }

    #[test]
    fn test_d4_restriction() {
        // Valid: d₄ in range 0-2
        assert!(claude_parse_unicode_escapes_manual("\\u{10000}").is_ok());
        assert!(claude_parse_unicode_escapes_manual("\\u{20000}").is_ok());

        // Invalid: d₄ > 2
        assert!(claude_parse_unicode_escapes_manual("\\u{30000}").is_err(),);
        assert!(claude_parse_unicode_escapes_manual("\\u{FFFFF}").is_err(),);
    }

    #[test]
    fn test_mixed_content() {
        let input = "Text with \\u0041 and \\u{1F44D} emoji";
        let expected = "Text with A and 👍 emoji";
        assert_eq!(
            claude_parse_unicode_escapes_manual(input).unwrap(),
            expected
        );
    }

    #[test]
    fn test_manual_parser() {
        assert_eq!(claude_parse_unicode_escapes_manual("\\u{41}").unwrap(), "A");
        assert_eq!(claude_parse_unicode_escapes_manual("\\u0041").unwrap(), "A");
        assert_eq!(
            claude_parse_unicode_escapes_manual("Hello \\u{1F600}!").unwrap(),
            "Hello 😀!"
        );
    }

    #[test]
    fn problem_id_12764() -> anyhow::Result<()> {
        let solution = r#"Strip-Player\u{1b}MyAgentBCDEFGH/newsurfer4/\u{a}"#;
        let expected = "Strip-Player\u{1b}MyAgentBCDEFGH/newsurfer4/\u{a}";

        assert_eq!(claude_parse_unicode_escapes_manual(solution)?, expected);
        Ok(())
    }
}
