use super::parse_session_config_file_with_computation;
use crate::choreography::{NetworkingStrategy, StorageStrategy};
use crate::execution::ExecutionContext;
use moose::prelude::*;
use moose::tokio;
use notify::{DebouncedEvent, Watcher};
use std::collections::HashMap;
use std::path::Path;

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
