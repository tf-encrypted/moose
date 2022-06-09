//! Choreography extensions.

pub mod filesystem;
pub mod grpc;

use crate::execution::{AsyncNetworkingImpl, AsyncStorageImpl};
use crate::prelude::SessionId;

pub type NetworkingStrategy = Box<dyn Fn(SessionId) -> AsyncNetworkingImpl + Send + Sync>;

pub type StorageStrategy = Box<dyn Fn() -> AsyncStorageImpl + Send + Sync>;
