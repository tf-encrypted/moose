//! Choreography extensions.

pub mod filesystem;
pub mod grpc;

use moose::execution::{AsyncNetworkingImpl, AsyncStorageImpl, Identity};
use moose::prelude::{Computation, Role, SessionId};
use serde::Deserialize;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::Path;
use std::str::FromStr;

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
    pub format: ComputationFormat,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputationFormat {
    Binary,
    Textual,
    Bincode,
}

#[derive(Debug, Deserialize)]
pub struct RoleConfig {
    pub name: String,
    pub endpoint: String,
}

pub fn parse_session_config_file_with_computation(
    session_config_file: &Path,
) -> Result<(SessionConfig, SessionId, RoleAssignment, Computation), Box<dyn std::error::Error>> {
    let (session_config, session_id, role_assignment) =
        parse_session_config_file_without_computation(session_config_file)?;

    let computation = {
        let comp_path = &session_config.computation.path;
        match session_config.computation.format {
            ComputationFormat::Binary => {
                let comp_raw = std::fs::read(comp_path)?;
                moose::computation::Computation::from_msgpack(comp_raw)?
            }
            ComputationFormat::Textual => {
                let comp_raw = std::fs::read_to_string(comp_path)?;
                moose::computation::Computation::from_textual(&comp_raw)?
            }
            ComputationFormat::Bincode => {
                let comp_raw = std::fs::read(comp_path)?;
                moose::computation::Computation::from_bincode(&comp_raw)?
            }
        }
    };

    Ok((session_config, session_id, role_assignment, computation))
}

pub fn parse_session_config_file_without_computation(
    session_config_file: &Path,
) -> Result<(SessionConfig, SessionId, RoleAssignment), Box<dyn std::error::Error>> {
    let session_config = SessionConfig::from_str(&std::fs::read_to_string(session_config_file)?)?;

    let role_assignment: RoleAssignment = session_config
        .roles
        .iter()
        .map(|role_config| {
            let role = Role::from(&role_config.name);
            let identity = Identity::from(&role_config.endpoint);
            (role, identity)
        })
        .collect();

    let session_id: SessionId = SessionId::try_from(
        session_config_file
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .borrow(),
    )?;

    Ok((session_config, session_id, role_assignment))
}

pub type RoleAssignment = HashMap<Role, Identity>;
