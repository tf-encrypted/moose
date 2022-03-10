//! Constants for the exponential backoff policy for gRPC
use lazy_static::lazy_static;
use std::time::Duration;

/// The default multiplier value
pub(crate) const MULTIPLIER: f64 = 1.;

lazy_static! {
    /// The default maximum elapsed time in milliseconds (1 minute).
    pub(crate) static ref MAX_ELAPSED_TIME: Option<Duration> = Some(Duration::from_millis(60_000));
    /// The default maximum back off time in milliseconds (3600 seconds).
    pub(crate) static ref MAX_INTERVAL: Duration = Duration::from_millis(3_600_000);
}
