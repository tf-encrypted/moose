use crate::computation::Ty;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, thiserror::Error)]
pub enum Error {
    #[error("Unexpected error")]
    Unexpected,

    #[error("Kernel operand was unavailable")]
    OperandUnavailable,

    #[error("Kernel result is unused")]
    ResultUnused,

    #[error("Type mismatch, expected {expected} but found {found}")]
    TypeMismatch { expected: String, found: Ty },

    #[error("Operator instantiation not supported: {0}")]
    UnimplementedOperator(String),

    #[error("Missing argument '{0}'")]
    MissingArgument(String),

    #[error("Malformed computation: operand '{0}' not found")]
    MalformedEnvironment(String),

    #[error("Malformed computation")]
    MalformedComputation(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Networking error: {0}")]
    Networking(String),

    #[error("Storage error: {0}")]
    Storage(String),
}

pub type Result<T> = std::result::Result<T, Error>;
