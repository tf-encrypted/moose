//! Storage implementation for local (debugging) execution.

use super::*;
use std::collections::HashMap;

#[derive(Default)]
pub struct LocalSyncStorage {
    store: std::sync::RwLock<HashMap<String, Value>>,
}

impl LocalSyncStorage {
    pub fn from_hashmap(h: HashMap<String, Value>) -> Self {
        LocalSyncStorage {
            store: std::sync::RwLock::new(h),
        }
    }
}

impl SyncStorage for LocalSyncStorage {
    fn save(&self, key: &str, _session_id: &SessionId, val: &Value) -> Result<()> {
        let mut store = self.store.write().map_err(|e| {
            tracing::error!("failed to get write lock: {e:?}");
            Error::Unexpected(None)
        })?;
        store.insert(key.to_string(), val.clone());
        Ok(())
    }

    fn load(
        &self,
        key: &str,
        _session_id: &SessionId,
        type_hint: Option<Ty>,
        query: &str,
    ) -> Result<Value> {
        match query {
            "" => Ok(()),
            _ => Err(Error::Storage(
                "query is not allowed for local storage".into(),
            )),
        }?;
        let store = self.store.read().map_err(|e| {
            tracing::error!("failed to get read lock: {e:?}");
            Error::Unexpected(None)
        })?;
        let item = store
            .get(key)
            .cloned()
            .ok_or_else(|| Error::Storage("key not found in store".into()))?;
        check_types(&item, &type_hint)?;
        Ok(item)
    }
}

#[derive(Default)]
pub struct LocalAsyncStorage {
    store: tokio::sync::RwLock<HashMap<String, Value>>,
}

impl LocalAsyncStorage {
    pub fn from_hashmap(h: HashMap<String, Value>) -> Self {
        LocalAsyncStorage {
            store: tokio::sync::RwLock::new(h),
        }
    }
}

#[async_trait]
impl AsyncStorage for LocalAsyncStorage {
    async fn save(&self, key: &str, _session_id: &SessionId, val: &Value) -> Result<()> {
        tracing::debug!("Async storage saving; key:'{key}'");
        let mut store = self.store.write().await;
        store.insert(key.to_string(), val.clone());
        Ok(())
    }

    async fn load(
        &self,
        key: &str,
        _session_id: &SessionId,
        type_hint: Option<Ty>,
        query: &str,
    ) -> Result<Value> {
        tracing::debug!("Async storage loading; key:'{key}'");
        match query {
            "" => Ok(()),
            _ => Err(Error::Storage(
                "query is not allowed for local storage".into(),
            )),
        }?;
        let store = self.store.read().await;
        let item = store
            .get(key)
            .cloned()
            .ok_or_else(|| Error::Storage("key not found in store".into()))?;
        check_types(&item, &type_hint)?;
        Ok(item)
    }
}

fn check_types(item: &Value, type_hint: &Option<Ty>) -> Result<()> {
    let item_ty = item.ty();
    match type_hint {
        Some(ty) => {
            if item_ty == *ty {
                Ok(())
            } else {
                Err(Error::Storage(format!(
                    "type hint does not match type of item: type_hint: {type_hint:?} type of item: {item_ty:?}"
                )))
            }
        }
        None => Ok(()),
    }
}
