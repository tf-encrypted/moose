mod gen {
    tonic::include_proto!("moose");
}

use self::gen::networking_client::NetworkingClient;
use self::gen::networking_server::Networking;
use self::gen::networking_server::NetworkingServer;
use self::gen::{SendValueRequest, SendValueResponse};
use crate::networking::constants;
use async_cell::sync::AsyncCell;
use async_trait::async_trait;
use backoff::future::retry;
use backoff::ExponentialBackoff;
use dashmap::DashMap;
use moose::networking::AsyncNetworking;
use moose::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tonic::transport::{Channel, Uri};

#[derive(Default, Clone)]
pub struct GrpcNetworkingManager {
    stores: Arc<SessionStores>,
    channels: Arc<Channels>,
}

impl GrpcNetworkingManager {
    pub fn new_server(&self) -> NetworkingServer<impl Networking> {
        NetworkingServer::new(NetworkingImpl {
            stores: Arc::clone(&self.stores),
        })
    }

    pub fn new_session(&self, session_id: SessionId) -> Arc<impl AsyncNetworking> {
        Arc::new(GrpcNetworking {
            session_id,
            stores: Arc::clone(&self.stores),
            channels: Arc::clone(&self.channels),
        })
    }
}

struct GrpcNetworking {
    session_id: SessionId,
    stores: Arc<SessionStores>,
    channels: Arc<Channels>,
}

impl GrpcNetworking {
    fn channel(&self, receiver: &Identity) -> moose::Result<Channel> {
        let channel = self
            .channels
            .entry(receiver.clone())
            .or_try_insert_with(|| {
                tracing::debug!("Creating channel to '{}'", receiver);
                let endpoint: Uri = format!("http://{}", receiver).parse().map_err(|_e| {
                    moose::Error::Networking(format!(
                        "failed to parse identity as endpoint: {:?}",
                        receiver
                    ))
                })?;
                Ok(Channel::builder(endpoint).connect_lazy())
            })?
            .clone(); // cloning channels is cheap per tonic documentation
        Ok(channel)
    }
}

#[async_trait]
impl AsyncNetworking for GrpcNetworking {
    async fn send(
        &self,
        val: &Value,
        receiver: &Identity,
        rendezvous_key: &RendezvousKey,
        _session_id: &SessionId,
    ) -> moose::Result<()> {
        retry(
            ExponentialBackoff {
                max_elapsed_time: *constants::MAX_ELAPSED_TIME,
                max_interval: *constants::MAX_INTERVAL,
                multiplier: constants::MULTIPLIER,
                ..Default::default()
            },
            || async {
                let tagged_value = TaggedValue {
                    session_id: self.session_id.clone(),
                    rendezvous_key: rendezvous_key.clone(),
                    value: val.clone(),
                };
                let bytes = bincode::serialize(&tagged_value)
                    .map_err(|e| moose::Error::Networking(e.to_string()))?;
                let request = SendValueRequest {
                    tagged_value: bytes,
                };
                let channel = self.channel(receiver)?;
                let mut client = NetworkingClient::new(channel);
                let _response = client
                    .send_value(request)
                    .await
                    .map_err(|e| moose::Error::Networking(e.to_string()))?;
                Ok(())
            },
        )
        .await
    }

    async fn receive(
        &self,
        _sender: &Identity,
        rendezvous_key: &RendezvousKey,
        _session_id: &SessionId,
    ) -> moose::Result<Value> {
        let cell = cell(
            &self.stores,
            self.session_id.clone(),
            rendezvous_key.clone(),
        );
        Ok(cell.take().await)
    }
}

impl Drop for GrpcNetworking {
    fn drop(&mut self) {
        let _ = self.stores.remove(&self.session_id);
    }
}

type SessionStore = DashMap<RendezvousKey, Arc<AsyncCell<Value>>>;

type SessionStores = DashMap<SessionId, Arc<SessionStore>>;

type Channels = DashMap<Identity, Channel>;

#[derive(Default)]
struct NetworkingImpl {
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

#[derive(Serialize, Deserialize)]
struct TaggedValue {
    session_id: SessionId,
    rendezvous_key: RendezvousKey,
    value: Value,
}
