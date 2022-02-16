use moose::execution::{AsyncNetworkingImpl, AsyncStorageImpl};

pub mod filesystem;

pub type NetworkingStrategy = Box<dyn Fn() -> AsyncNetworkingImpl>;

pub type StorageStrategy = Box<dyn Fn() -> AsyncStorageImpl>;
