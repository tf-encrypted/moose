use moose::computation::SessionId;
use moose::execution::{AsyncNetworkingImpl, AsyncStorageImpl};

pub type NetworkingStrategy = Box<dyn Fn(SessionId) -> AsyncNetworkingImpl + Send + Sync>;

pub type StorageStrategy = Box<dyn Fn() -> AsyncStorageImpl + Send + Sync>;

pub mod filesystem;
pub mod grpc;
