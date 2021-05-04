use crate::computation::*;
use std::convert::TryFrom;

impl TryFrom<&str> for Computation {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> anyhow::Result<Computation> {
        Err(anyhow::anyhow!("Not implemented yet {}", value))
    }
}
