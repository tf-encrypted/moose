//! Filesystem-based choreography.

use crate::choreography::{NetworkingStrategy, StorageStrategy};
use crate::computation::Computation;
use crate::execution::ExecutionContext;
use crate::execution::RoleAssignment;
use crate::prelude::*;
use notify::{DebouncedEvent, Watcher};
use serde::Deserialize;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::Path;
use std::str::FromStr;

/// Filesystem-based choreography.
///
/// Sessions are controlled by two sets of files:
///
/// - `.moose` files that contain Moose computations;
///   these must be physical computations.
///
/// - `.session` files that specify session configurations;
///   the name of the file is used to derive the session id.
///
/// `FilesystemChoreography` listens for changes to the sessions directory
/// and will launch new sessions when new `.session` files are created.
pub struct FilesystemChoreography {
    own_identity: Identity,
    sessions_dir: String,
    networking_strategy: NetworkingStrategy,
    storage_strategy: StorageStrategy,
}

impl FilesystemChoreography {
    pub fn new(
        own_identity: Identity,
        sessions_dir: String,
        networking_strategy: NetworkingStrategy,
        storage_strategy: StorageStrategy,
    ) -> FilesystemChoreography {
        FilesystemChoreography {
            own_identity,
            sessions_dir,
            networking_strategy,
            storage_strategy,
        }
    }

    #[tracing::instrument(skip(self, ignore_existing, no_listen))]
    pub async fn process(
        &self,
        ignore_existing: bool,
        no_listen: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !ignore_existing {
            for entry in std::fs::read_dir(&self.sessions_dir)? {
                let entry = entry?;
                let path = entry.path();
                self.launch_session_from_path(&path).await?;
            }
        }

        if !no_listen {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut watcher = notify::watcher(tx.clone(), std::time::Duration::from_secs(2))?;
            watcher.watch(&self.sessions_dir, notify::RecursiveMode::Recursive)?;

            for event in rx {
                match event {
                    DebouncedEvent::Create(path) => {
                        self.abort_session_from_path(&path).await?;
                        self.launch_session_from_path(&path).await?;
                    }
                    DebouncedEvent::Remove(path) => {
                        self.abort_session_from_path(&path).await?;
                    }
                    DebouncedEvent::Write(path) => {
                        self.abort_session_from_path(&path).await?;
                        self.launch_session_from_path(&path).await?;
                    }
                    DebouncedEvent::Rename(src_path, dst_path) => {
                        self.abort_session_from_path(&src_path).await?;
                        self.launch_session_from_path(&dst_path).await?;
                    }
                    _ => {
                        // ignore
                    }
                }
            }
        }

        Ok(())
    }

    async fn launch_session_from_path(
        &self,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if path.is_file() {
            match path.extension() {
                Some(ext) if ext == "session" => {
                    let session_handle = self.launch_session(path).await?;
                    let res = session_handle.join_on_first_error().await;
                    if let Err(e) = res {
                        tracing::error!("Session error: {}", e);
                        return Result::Err(e.into());
                    }
                }
                Some(ext) if ext == "moose" => {
                    // ok to skip
                }
                _ => {
                    tracing::warn!("Skipping {:?}", path);
                }
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip(self, path))]
    async fn launch_session(
        &self,
        path: &Path,
    ) -> Result<AsyncSessionHandle, Box<dyn std::error::Error>> {
        tracing::info!("Loading session from {:?}", path);
        let (_, session_id, role_assignments, computation) =
            parse_session_config_file_with_computation(path)?;
        let networking = (self.networking_strategy)(session_id.clone());
        let storage = (self.storage_strategy)();

        let context = ExecutionContext::new(self.own_identity.clone(), networking, storage);

        // TODO(Morten) for now we don't support arguments in this type of choreography;
        // we could be eg allowing them to be specified in the .session file (perhaps as
        // names of .npy files)
        let arguments = HashMap::new();

        tracing::debug!("Scheduling computation");
        let (handle, outputs) = context
            .execute_indexed_computation(
                session_id.clone(),
                &computation,
                arguments,
                role_assignments,
            )
            .await?;

        tracing::debug!("Ready for outputs");
        for (output_name, output_value) in outputs {
            let session_id = session_id.clone();
            tokio::spawn(async move {
                let value = output_value.await.unwrap();
                tracing::info!(
                    "Output '{}' from '{}' ready: {:?}",
                    output_name,
                    session_id,
                    value
                );
            });
        }

        Ok(handle)
    }

    async fn abort_session_from_path(
        &self,
        _path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // TODO
        Ok(())
    }
}

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
                Computation::from_msgpack(&comp_raw)?
            }
            ComputationFormat::Textual => {
                let comp_raw = std::fs::read_to_string(comp_path)?;
                Computation::from_textual(&comp_raw)?
            }
            ComputationFormat::Bincode => {
                let comp_raw = std::fs::read(comp_path)?;
                Computation::from_bincode(&comp_raw)?
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
