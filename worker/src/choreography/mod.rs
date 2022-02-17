use moose::execution::{AsyncNetworkingImpl, AsyncStorageImpl};

pub type NetworkingStrategy = Box<dyn Fn() -> AsyncNetworkingImpl>;

pub type StorageStrategy = Box<dyn Fn() -> AsyncStorageImpl>;

pub mod filesystem;
