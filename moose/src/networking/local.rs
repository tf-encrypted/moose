//! Networking implementation for local (debugging) execution.

use super::*;
use std::collections::HashMap;
use std::sync::Arc;

///
/// This implementation is intended for local development/testing purposes
/// only. It simply stores all values in a hashmap without any actual networking.
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
            Error::Unexpected(None)
        })?;
        if store.contains_key(&key) {
            tracing::error!("value has already been sent");
            return Err(Error::Unexpected(None));
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
            Error::Unexpected(None)
        })?;
        store.get(&key).cloned().ok_or_else(|| {
            tracing::error!("Key not found in store");
            Error::Unexpected(None)
        })
    }
}

/// A simple implementation of asynchronous networking for local execution.
///
/// This implementation is intended for local development/testing purposes
/// only. It simply stores all values in a hashmap without any actual networking.
#[derive(Default)]
pub struct LocalAsyncNetworking {
    store: dashmap::DashMap<String, Arc<async_cell::sync::AsyncCell<Value>>>,
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
        let cell = self
            .store
            .entry(key)
            .or_insert_with(async_cell::sync::AsyncCell::shared)
            .value()
            .clone();
        cell.set(val.clone());
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
        let cell = self
            .store
            .entry(key)
            .or_insert_with(async_cell::sync::AsyncCell::shared)
            .value()
            .clone();
        let val = cell.get().await;
        Ok(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::HostPlacement;
    use std::convert::TryInto;

    #[test]
    fn sync_networking() {
        let net = LocalSyncNetworking::default();

        let alice = "alice".into();
        let bob = "bob".into();

        let unit = Value::HostUnit(Box::new(HostUnit(HostPlacement::from("alice"))));

        net.send(
            &unit,
            &bob,
            &"rdv".try_into().unwrap(),
            &"12345".try_into().unwrap(),
        )
        .unwrap();
        net.send(
            &unit,
            &bob,
            &"rdv".try_into().unwrap(),
            &"67890".try_into().unwrap(),
        )
        .unwrap();
        net.receive(
            &alice,
            &"rdv".try_into().unwrap(),
            &"12345".try_into().unwrap(),
        )
        .unwrap();
        net.receive(
            &alice,
            &"rdv".try_into().unwrap(),
            &"67890".try_into().unwrap(),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn async_networking() {
        use std::sync::Arc;

        let net = Arc::new(LocalAsyncNetworking::default());

        let net1 = Arc::clone(&net);
        let task1 = tokio::spawn(async move {
            let alice = "alice".into();
            net1.receive(
                &alice,
                &"rdv".try_into().unwrap(),
                &"12345".try_into().unwrap(),
            )
            .await
        });

        let net2 = Arc::clone(&net);
        let task2 = tokio::spawn(async move {
            let alice = "alice".into();
            net2.receive(
                &alice,
                &"rdv".try_into().unwrap(),
                &"67890".try_into().unwrap(),
            )
            .await
        });

        let net3 = Arc::clone(&net);
        let task3 = tokio::spawn(async move {
            let bob = "bob".into();
            let unit = Value::HostUnit(Box::new(HostUnit(HostPlacement::from("alice"))));
            net3.send(
                &unit,
                &bob,
                &"rdv".try_into().unwrap(),
                &"12345".try_into().unwrap(),
            )
            .await
        });

        let net4 = Arc::clone(&net);
        let task4 = tokio::spawn(async move {
            let bob = "bob".into();
            let unit = Value::HostUnit(Box::new(HostUnit(HostPlacement::from("alice"))));
            net4.send(
                &unit,
                &bob,
                &"rdv".try_into().unwrap(),
                &"67890".try_into().unwrap(),
            )
            .await
        });

        let _ = tokio::try_join!(task1, task2, task3, task4).unwrap();
    }
}
