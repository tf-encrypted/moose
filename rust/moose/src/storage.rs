use crate::computation::*;
use crate::error::{Error, Result};
use async_trait::async_trait;
use std::collections::HashMap;

pub trait SyncStorage {
    fn save(&self, key: String, val: Value) -> Result<()>;
    fn load(&self, key: String) -> Result<Value>;
}

#[async_trait]
pub trait AsyncStorage {
    async fn save(&self, key: String, val: Value) -> Result<()>;

    async fn load(&self, key: String) -> Result<Value>;
}

#[derive(Default)]
pub struct LocalSyncStorage {
    store: std::sync::RwLock<HashMap<String, Value>>,
}

impl SyncStorage for LocalSyncStorage {
    fn save(&self, key: String, val: Value) -> Result<()> {
        let mut store = self.store.write().map_err(|e| {
            tracing::error!("failed to get write lock: {:?}", e);
            Error::Unexpected
        })?;
        if store.contains_key(&key) {
            tracing::error!("value has already been sent");
            return Err(Error::Unexpected);
        }
        store.insert(key, val.clone());
        Ok(())
    }

    fn load(&self, key: String) -> Result<Value> {
        let store = self.store.read().map_err(|e| {
            tracing::error!("failed to get read lock: {:?}", e);
            Error::Unexpected
        })?;
        store.get(&key).cloned().ok_or_else(|| {
            tracing::error!("Key not found in store");
            Error::Unexpected
        })
    }
}

#[derive(Default)]
pub struct LocalAsyncStorage {
    store: tokio::sync::RwLock<HashMap<String, Value>>,
}

#[async_trait]
impl AsyncStorage for LocalAsyncStorage {
    async fn save(&self, key: String, val: Value) -> Result<()> {
        tracing::debug!("Async storage saving; key:'{}'", key);
        let mut store = self.store.write().await;
        if store.contains_key(&key) {
            tracing::error!("value has already been sent");
            return Err(Error::Unexpected);
        }
        store.insert(key, val.clone());
        Ok(())
    }

    async fn load(&self, key: String) -> Result<Value> {
        tracing::debug!("Async storage loading; key:'{}'", key,);
        loop {
            {
                let store = self.store.read().await;
                if let Some(val) = store.get(&key).cloned() {
                    return Ok(val);
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }
}
