//! Moose error types.

use crate::computation::Ty;
use serde::{Deserialize, Serialize};

/// Common error type used throughout.
#[derive(Clone, Debug, Deserialize, Serialize, thiserror::Error)]
pub enum Error {
    #[error("Unexpected error")]
    Unexpected(Option<String>),

    #[error("Kernel operand was unavailable")]
    OperandUnavailable,

    #[error("Kernel result is unused")]
    ResultUnused,

    #[error("Type mismatch, expected {expected} but found {found}")]
    TypeMismatch { expected: String, found: Ty },

    #[error("Operator instantiation not supported: {0}")]
    UnimplementedOperator(String),

    #[error("Kernel error: {0}")]
    KernelError(String),

    #[error("Missing argument '{0}'")]
    MissingArgument(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Malformed computation: operand '{0}' not found")]
    MalformedEnvironment(String),

    #[error("Malformed computation")]
    MalformedComputation(String),

    #[error("Malformed placement")]
    MalformedPlacement,

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Networking error: {0}")]
    Networking(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Test runtime error: {0}")]
    TestRuntime(String),

    #[error("Session {0} already exists for this executor")]
    SessionAlreadyExists(String),

    #[error("Failed to serialize computation: {0}")]
    SerializationError(String),
}

pub type Result<T> = std::result::Result<T, Error>;
