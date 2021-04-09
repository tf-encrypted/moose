use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, thiserror::Error)]
pub enum Error {
    #[error("Unexpected error")]
    Unexpected,

    #[error("Input to kernel unavailable")]
    InputUnavailable,

    #[error("Type mismatch")]
    TypeMismatch,

    #[error("Operator type instantiation not supported: {0}")]
    TypeMismatchOperator(String),

    #[error("Operator instantiation not supported: {0}")]
    UnimplementedOperator(String),

    #[error("Malformed environment")]
    MalformedEnvironment,

    #[error("Malformed computation")]
    MalformedComputation(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Networking error: {0}")]
    Networking(String),
}

pub type Result<T> = std::result::Result<T, Error>;
