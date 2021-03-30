use crate::computation::*;
use crate::error::{Error, Result};
use async_trait::async_trait;
use std::collections::HashMap;

pub trait SyncNetworking {
    fn send(&self, v: &Value, rendezvous_key: &RendezvousKey, session_id: &SessionId)
        -> Result<()>;
    fn receive(&self, rendezvous_key: &RendezvousKey, session_id: &SessionId) -> Result<Value>;
}

#[async_trait]
pub trait AsyncNetworking {
    async fn send(
        &self,
        v: &Value,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()>;
    async fn receive(
        &self,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value>;
}

#[derive(Default)]
pub struct LocalSyncNetworking {
    store: std::sync::RwLock<HashMap<String, Value>>,
}

impl SyncNetworking for LocalSyncNetworking {
    fn send(
        &self,
        val: &Value,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()> {
        let key = format!("{}/{}", session_id, rendezvous_key);
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

    fn receive(&self, rendezvous_key: &RendezvousKey, session_id: &SessionId) -> Result<Value> {
        let key = format!("{}/{}", session_id, rendezvous_key);
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
pub struct LocalAsyncNetworking {
    store: tokio::sync::RwLock<HashMap<String, Value>>,
}

#[async_trait]
impl AsyncNetworking for LocalAsyncNetworking {
    async fn send(
        &self,
        val: &Value,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()> {
        tracing::debug!("Async sending; rdv:'{}' sid:{}", rendezvous_key, session_id);
        let key = format!("{}/{}", session_id, rendezvous_key);
        let mut store = self.store.write().await;
        if store.contains_key(&key) {
            tracing::error!("value has already been sent");
            return Err(Error::Unexpected);
        }
        store.insert(key, val.clone());
        Ok(())
    }

    async fn receive(
        &self,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value> {
        tracing::debug!(
            "Async receiving; rdv:'{}', sid:{}",
            rendezvous_key,
            session_id
        );
        loop {
            let key = format!("{}/{}", session_id, rendezvous_key);
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

pub struct DummyNetworking(pub Value);

impl SyncNetworking for DummyNetworking {
    fn send(
        &self,
        _val: &Value,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()> {
        tracing::debug!("Sending; rdv:'{}' sid:{}", rendezvous_key, session_id);
        Ok(())
    }

    fn receive(&self, rendezvous_key: &RendezvousKey, session_id: &SessionId) -> Result<Value> {
        tracing::debug!("Receiving; rdv:'{}', sid:{}", rendezvous_key, session_id);
        Ok(self.0.clone())
    }
}

#[async_trait]
impl AsyncNetworking for DummyNetworking {
    async fn send(
        &self,
        _val: &Value,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()> {
        tracing::debug!("Sending; rdv:'{}' sid:{}", rendezvous_key, session_id);
        Ok(())
    }

    async fn receive(
        &self,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value> {
        tracing::debug!("Receiving; rdv:'{}', sid:{}", rendezvous_key, session_id);
        Ok(self.0.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_networking() {
        let net = LocalSyncNetworking::default();

        net.send(&Value::Unit, "rdv", &12345).unwrap();
        net.send(&Value::Unit, "rdv", &67890).unwrap();
        net.receive("rdv", &12345).unwrap();
        net.receive("rdv", &67890).unwrap();
    }

    #[tokio::test]
    async fn async_networking() {
        use std::sync::Arc;

        let net = Arc::new(LocalAsyncNetworking::default());

        let net1 = Arc::clone(&net);
        let task1 = tokio::spawn(async move { net1.receive("rdv", &12345).await });

        let net2 = Arc::clone(&net);
        let task2 = tokio::spawn(async move { net2.receive("rdv", &67890).await });

        let net3 = Arc::clone(&net);
        let task3 = tokio::spawn(async move { net3.send(&Value::Unit, "rdv", &12345).await });

        let net4 = Arc::clone(&net);
        let task4 = tokio::spawn(async move { net4.send(&Value::Unit, "rdv", &67890).await });

        let _ = tokio::try_join!(task1, task2, task3, task4).unwrap();
    }
}
