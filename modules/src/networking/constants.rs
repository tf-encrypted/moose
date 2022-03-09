//! Constants for the exponential backoff policy for gRPC

/// The default multiplier value
pub(crate) const MULTIPLIER: f64 = 1.;
/// The default maximum back off time in milliseconds (3600 seconds).
pub(crate) const MAX_INTERVAL_MILLIS: u64 = 3_600_000;
/// The default maximum elapsed time in milliseconds (1 minute).
pub(crate) const MAX_ELAPSED_TIME_MILLIS: u64 = 1_000;
