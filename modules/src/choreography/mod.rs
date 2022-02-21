use moose::computation::SessionId;
use moose::execution::{AsyncNetworkingImpl, AsyncStorageImpl};

pub type NetworkingStrategy = Box<dyn Fn(SessionId) -> AsyncNetworkingImpl>;

pub type StorageStrategy = Box<dyn Fn() -> AsyncStorageImpl>;

pub mod filesystem;
