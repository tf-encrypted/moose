use std::str::FromStr;

use moose::computation::SessionId;
use moose::execution::{AsyncNetworkingImpl, AsyncStorageImpl};
use serde::Deserialize;

pub type NetworkingStrategy = Box<dyn Fn(SessionId) -> AsyncNetworkingImpl + Send + Sync>;

pub type StorageStrategy = Box<dyn Fn() -> AsyncStorageImpl + Send + Sync>;

#[derive(Debug, Deserialize)]
pub struct SessionConfig {
    pub computation: ComputationConfig,
    pub roles: Vec<RoleConfig>,
}

impl FromStr for SessionConfig {
    type Err = toml::de::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        toml::from_str(s)
    }
}

#[derive(Debug, Deserialize)]
pub struct ComputationConfig {
    pub path: String,
    pub format: Format,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Binary,
    Textual,
}

#[derive(Debug, Deserialize)]
pub struct RoleConfig {
    pub name: String,
    pub endpoint: String,
}

pub mod filesystem;
pub mod grpc;
