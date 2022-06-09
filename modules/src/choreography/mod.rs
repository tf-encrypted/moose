//! Choreography extensions.

pub mod filesystem;
pub mod grpc;

use moose::execution::{AsyncNetworkingImpl, AsyncStorageImpl};
use moose::prelude::SessionId;

pub type NetworkingStrategy = Box<dyn Fn(SessionId) -> AsyncNetworkingImpl + Send + Sync>;

pub type StorageStrategy = Box<dyn Fn() -> AsyncStorageImpl + Send + Sync>;
