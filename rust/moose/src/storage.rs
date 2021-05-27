use crate::computation::*;
use crate::error::{Error, Result};
use async_trait::async_trait;
use std::collections::HashMap;

pub trait SyncStorage {
    fn save(&self, key: &str, val: &Value) -> Result<()>;
    fn load(&self, key: &str, type_hint: Option<Ty>, query: Option<String>) -> Result<Value>;
}

#[async_trait]
pub trait AsyncStorage {
    async fn save(&self, key: &str, val: &Value) -> Result<()>;
    async fn load(&self, key: &str, type_hint: Option<Ty>, query: Option<String>) -> Result<Value>;
}

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
    fn save(&self, key: &str, val: &Value) -> Result<()> {
        let mut store = self.store.write().map_err(|e| {
            tracing::error!("failed to get write lock: {:?}", e);
            Error::Unexpected
        })?;
        store.insert(key.to_string(), val.clone());
        Ok(())
    }

    fn load(&self, key: &str, type_hint: Option<Ty>, query: Option<String>) -> Result<Value> {
        match query {
            None => Ok(()),
            _ => Err(Error::Storage(
                "query is not allowed for local storage".into(),
            )),
        }?;
        let store = self.store.read().map_err(|e| {
            tracing::error!("failed to get read lock: {:?}", e);
            Error::Unexpected
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
    async fn save(&self, key: &str, val: &Value) -> Result<()> {
        tracing::debug!("Async storage saving; key:'{}'", key);
        let mut store = self.store.write().await;
        store.insert(key.to_string(), val.clone());
        Ok(())
    }

    async fn load(&self, key: &str, type_hint: Option<Ty>, query: Option<String>) -> Result<Value> {
        tracing::debug!("Async storage loading; key:'{}'", key,);
        match query {
            None => Ok(()),
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
    let item_ty = value_ty(&item)?;
    match type_hint {
        Some(ty) => {
            if item_ty == *ty {
                Ok(())
            } else {
                Err(Error::Storage(
                    format!(
                        "type hint does not match type of item: type_hint: {:?} type of item: {:?}",
                        type_hint, item_ty
                    )
                    .into(),
                ))
            }
        }
        None => Ok(()),
    }
}

fn value_ty(val: &Value) -> Result<Ty> {
    match val {
        Value::Float64Tensor(_) => Ok(Ty::Float64TensorTy),
        Value::Float32Tensor(_) => Ok(Ty::Float32TensorTy),
        Value::Int32Tensor(_) => Ok(Ty::Int32TensorTy),
        Value::Int64Tensor(_) => Ok(Ty::Int64TensorTy),
        Value::Uint64Tensor(_) => Ok(Ty::Uint64TensorTy),
        Value::Uint32Tensor(_) => Ok(Ty::Uint32TensorTy),
        _ => Err(Error::Storage("variant not implemented".into())),
    }
}
