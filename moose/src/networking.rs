use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::Identity;
use async_trait::async_trait;
use std::collections::HashMap;

pub trait SyncNetworking {
    fn send(
        &self,
        value: &Value,
        receiver: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()>;
    fn receive(
        &self,
        sender: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value>;
}

#[async_trait]
pub trait AsyncNetworking {
    async fn send(
        &self,
        value: &Value,
        receiver: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()>;
    async fn receive(
        &self,
        sender: &Identity,
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
        _receiver: &Identity,
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

    fn receive(
        &self,
        _sender: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value> {
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
        _receiver: &Identity,
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
        _sender: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value> {
        tracing::debug!(
            "Async receiving; rdv:'{}', sid:{}",
            rendezvous_key,
            session_id
        );
        let key = format!("{}/{}", session_id, rendezvous_key);
        // note that we are using a loop since the store doesn't immediately
        // allow us to block until a value is present
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

pub struct DummyNetworking(pub Value);

impl SyncNetworking for DummyNetworking {
    fn send(
        &self,
        _value: &Value,
        _receiver: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()> {
        tracing::debug!("Sending; rdv:'{}' sid:{}", rendezvous_key, session_id);
        Ok(())
    }

    fn receive(
        &self,
        _sender: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value> {
        tracing::debug!("Receiving; rdv:'{}', sid:{}", rendezvous_key, session_id);
        Ok(self.0.clone())
    }
}

#[async_trait]
impl AsyncNetworking for DummyNetworking {
    async fn send(
        &self,
        _value: &Value,
        _receiver: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()> {
        tracing::debug!("Sending; rdv:'{}' sid:{}", rendezvous_key, session_id);
        Ok(())
    }

    async fn receive(
        &self,
        _sender: &Identity,
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
    use std::convert::TryFrom;

    #[test]
    fn sync_networking() {
        let net = LocalSyncNetworking::default();

        let alice = "alice".into();
        let bob = "bob".into();

        let unit = Value::Unit(Unit(HostPlacement {
            owner: "alice".into(),
        }));

        net.send(&unit, &bob, "rdv", &SessionId::try_from("12345").unwrap())
            .unwrap();
        net.send(&unit, &bob, "rdv", &SessionId::try_from("67890").unwrap())
            .unwrap();
        net.receive(&alice, "rdv", &SessionId::try_from("12345").unwrap())
            .unwrap();
        net.receive(&alice, "rdv", &SessionId::try_from("67890").unwrap())
            .unwrap();
    }

    #[tokio::test]
    async fn async_networking() {
        use std::sync::Arc;

        let net = Arc::new(LocalAsyncNetworking::default());

        let net1 = Arc::clone(&net);
        let task1 = tokio::spawn(async move {
            let alice = "alice".into();
            net1.receive(&alice, "rdv", &SessionId::try_from("12345").unwrap())
                .await
        });

        let net2 = Arc::clone(&net);
        let task2 = tokio::spawn(async move {
            let alice = "alice".into();
            net2.receive(&alice, "rdv", &SessionId::try_from("67890").unwrap())
                .await
        });

        let net3 = Arc::clone(&net);
        let task3 = tokio::spawn(async move {
            let bob = "bob".into();
            let unit = Value::Unit(Unit(HostPlacement {
                owner: "alice".into(),
            }));
            net3.send(&unit, &bob, "rdv", &SessionId::try_from("12345").unwrap())
                .await
        });

        let net4 = Arc::clone(&net);
        let task4 = tokio::spawn(async move {
            let bob = "bob".into();
            let unit = Value::Unit(Unit(HostPlacement {
                owner: "alice".into(),
            }));
            net4.send(&unit, &bob, "rdv", &SessionId::try_from("67890").unwrap())
                .await
        });

        let _ = tokio::try_join!(task1, task2, task3, task4).unwrap();
    }
}
