use crate::choreography::{NetworkingStrategy, StorageStrategy};
use crate::execution::ExecutionContext;
use moose::prelude::*;
use notify::{DebouncedEvent, Watcher};
use std::collections::HashMap;
use std::convert::TryFrom;

mod config;
use self::config::*;

pub struct FilesystemChoreography {
    own_identity: Identity,
    sessions_dir: String,
    // sessions: HashMap<SessionId, AsyncSession>,
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
        // let sessions = HashMap::new();
        FilesystemChoreography {
            own_identity,
            sessions_dir,
            // sessions,
            networking_strategy,
            storage_strategy,
        }
    }

    pub async fn listen(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::watcher(tx, std::time::Duration::from_secs(2))?;
        watcher.watch(&self.sessions_dir, notify::RecursiveMode::Recursive)?;

        let watcher_task = tokio::spawn(async move {
            for event in rx {
                tracing::debug!("Filesystem event: {:?}", event);
                match event {
                    DebouncedEvent::Create(_path) => {}
                    DebouncedEvent::Remove(_path) => {}
                    DebouncedEvent::Write(_path) => {}
                    DebouncedEvent::Rename(_src_path, _dst_path) => {}
                    _ => {
                        // ignore
                    }
                }
            }
        });
        tracing::debug!("File watcher launched");

        for entry in std::fs::read_dir(&self.sessions_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                match path.extension() {
                    Some(ext) if ext == "toml" => {
                        let config = std::fs::read_to_string(&path)?;
                        let session_config: SessionConfig = toml::from_str(&config)?;

                        let session_id = {
                            let raw_session_id =
                                path.file_stem().unwrap().to_string_lossy().to_string();
                            SessionId::try_from(raw_session_id.as_str()).unwrap()
                        };

                        let computation = {
                            let comp_path = &session_config.computation.path;
                            let comp_raw = std::fs::read_to_string(comp_path)?;
                            moose::textual::verbose_parse_computation(&comp_raw)?
                        };

                        let role_assignments = {
                            let mut role_assignments = HashMap::new();
                            for role_config in session_config.roles {
                                let role = Role::from(&role_config.name);
                                let identity = Identity::from(&role_config.endpoint);

                                role_assignments.insert(role, identity.clone());
                            }
                            role_assignments
                        };

                        let networking = (self.networking_strategy)();
                        let storage = (self.storage_strategy)();
                        let session =
                            ExecutionContext::new(self.own_identity.clone(), networking, storage);
                        session
                            .execute_computation(session_id, &computation, role_assignments)
                            .await?;
                    }
                    Some(ext) if ext == "moose" => {
                        // ok to skip
                    }
                    _ => {
                        tracing::warn!("Skipping {:?}", path);
                    }
                }
            }
        }

        watcher_task.await?;

        Ok(())
    }
}
