use std::{
    borrow::Cow,
    fmt::{self},
    path::PathBuf,
    time::Duration,
};

use anyhow::Context;
use clap::ValueEnum;
use rand::{Rng, SeedableRng, seq::IndexedRandom};
use serde::{Deserialize, Serialize};

pub const SENTINEL_ID: u32 = 69420;
