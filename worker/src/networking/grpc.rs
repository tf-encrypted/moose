use crate::gen::networking_client::NetworkingClient;
use crate::gen::networking_server::Networking;
use crate::gen::{SendValueRequest, SendValueResponse};
use async_cell::sync::AsyncCell;
use async_trait::async_trait;
use dashmap::DashMap;
use moose::networking::AsyncNetworking;
use moose::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tonic::transport::Channel;

pub struct GrpcNetworking {
    pub stores: Arc<SessionStores>,
    pub channels: Arc<Channels>,
}

#[derive(Serialize, Deserialize)]
struct TaggedValue {
    session_id: SessionId,
    rendezvous_key: RendezvousKey,
    value: Value,
}

#[async_trait]
impl AsyncNetworking for GrpcNetworking {
    async fn send(
        &self,
        val: &Value,
        receiver: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> moose::Result<()> {
        let tagged_value = TaggedValue {
            session_id: session_id.clone(),
            rendezvous_key: rendezvous_key.clone(),
            value: val.clone(),
        };

        let request = SendValueRequest {
            tagged_value: bincode::serialize(&tagged_value)
                .map_err(|e| moose::Error::Networking(e.to_string()))?,
        };

        let channel = self
            .channels
            .get(receiver)
            .ok_or_else(|| {
                moose::Error::Networking(format!("Channel for '{}' not found", receiver))
            })?
            .clone(); // cloning channels is cheap per tonic documentation
        let mut client = NetworkingClient::new(channel);

        let _response = client
            .send_value(request)
            .await
            .map_err(|e| moose::Error::Networking(e.to_string()))?;

        Ok(())
    }

    async fn receive(
        &self,
        _sender: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> moose::Result<Value> {
        let cell = cell(&self.stores, session_id.clone(), rendezvous_key.clone());
        Ok(cell.take().await)
    }
}

pub type SessionStore = DashMap<RendezvousKey, Arc<AsyncCell<Value>>>;

pub type SessionStores = DashMap<SessionId, Arc<SessionStore>>;

pub type Channels = DashMap<Identity, Channel>;

#[derive(Default)]
pub struct NetworkingImpl {
    pub stores: Arc<SessionStores>,
}

fn cell(
    stores: &Arc<SessionStores>,
    session_id: SessionId,
    rendezvous_key: RendezvousKey,
) -> Arc<AsyncCell<Value>> {
    let session_store = stores
        .entry(session_id) // TODO(Morten) only use the secure bytes?
        .or_insert_with(Arc::default)
        .value()
        .clone();

    let cell = session_store
        .entry(rendezvous_key)
        .or_insert_with(AsyncCell::shared)
        .value()
        .clone();

    cell
}

#[async_trait]
impl Networking for NetworkingImpl {
    async fn send_value(
        &self,
        request: tonic::Request<SendValueRequest>,
    ) -> Result<tonic::Response<SendValueResponse>, tonic::Status> {
        let request = request.into_inner();
        let tagged_value = bincode::deserialize::<TaggedValue>(&request.tagged_value).unwrap(); // TODO error handling

        let cell = cell(
            &self.stores,
            tagged_value.session_id,
            tagged_value.rendezvous_key,
        );
        cell.set(tagged_value.value);

        Ok(tonic::Response::new(SendValueResponse::default()))
    }
}
