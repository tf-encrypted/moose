//! gRPC-based networking implementation.

mod gen {
    tonic::include_proto!("moose_networking");
}

use self::gen::networking_client::NetworkingClient;
use self::gen::networking_server::{Networking, NetworkingServer};
use self::gen::{SendValueRequest, SendValueResponse};
use crate::networking::constants;
use crate::networking::AsyncNetworking;
use crate::prelude::*;
use crate::Error;
use async_cell::sync::AsyncCell;
use async_trait::async_trait;
use backoff::future::retry;
use backoff::ExponentialBackoff;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tonic::transport::{Channel, ClientTlsConfig, Uri};

#[derive(Default, Clone)]
pub struct GrpcNetworkingManager {
    stores: Arc<SessionStores>,
    channels: Arc<Channels>,
    tls_client_config: Option<ClientTlsConfig>,
}

impl GrpcNetworkingManager {
    pub fn new_server(&self) -> NetworkingServer<impl Networking> {
        NetworkingServer::new(NetworkingImpl {
            stores: Arc::clone(&self.stores),
        })
    }

    pub fn without_tls() -> Self {
        GrpcNetworkingManager {
            stores: Default::default(),
            channels: Default::default(),
            tls_client_config: None,
        }
    }

    pub fn from_tls_config(client: ClientTlsConfig) -> Self {
        GrpcNetworkingManager {
            stores: Default::default(),
            channels: Default::default(),
            tls_client_config: Some(client),
        }
    }

    pub fn new_session(&self, session_id: SessionId) -> Arc<impl AsyncNetworking> {
        Arc::new(GrpcNetworking {
            session_id,
            stores: Arc::clone(&self.stores),
            channels: Arc::clone(&self.channels),
            tls_config: self.tls_client_config.clone(),
        })
    }
}

pub struct GrpcNetworking {
    tls_config: Option<ClientTlsConfig>,
    session_id: SessionId,
    stores: Arc<SessionStores>,
    channels: Arc<Channels>,
}

impl GrpcNetworking {
    fn channel(&self, receiver: &Identity) -> crate::Result<Channel> {
        let channel = self
            .channels
            .entry(receiver.clone())
            .or_try_insert_with(|| {
                tracing::debug!("Creating channel to '{}'", receiver);
                let endpoint: Uri = format!("http://{}", receiver).parse().map_err(|_e| {
                    Error::Networking(format!(
                        "failed to parse identity as endpoint: {:?}",
                        receiver
                    ))
                })?;

                let mut channel = Channel::builder(endpoint);
                if let Some(ref tls_config) = self.tls_config {
                    channel = channel.tls_config(tls_config.clone()).map_err(|e| {
                        Error::Networking(format!("failed to TLS config {:?}", e.to_string()))
                    })?;
                };
                Ok(channel.connect_lazy())
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
    ) -> crate::Result<()> {
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
                    .map_err(|e| Error::Networking(e.to_string()))?;
                let request = SendValueRequest {
                    tagged_value: bytes,
                };
                let channel = self.channel(receiver)?;
                let mut client = NetworkingClient::new(channel);
                #[cfg(debug_assertions)]
                tracing::debug!("Sending '{}' to {}", rendezvous_key, receiver);
                let _response = client
                    .send_value(request)
                    .await
                    .map_err(|e| Error::Networking(e.to_string()))?;
                Ok(())
            },
        )
        .await
    }

    async fn receive(
        &self,
        sender: &Identity,
        rendezvous_key: &RendezvousKey,
        _session_id: &SessionId,
    ) -> crate::Result<Value> {
        let cell = cell(
            &self.stores,
            self.session_id.clone(),
            rendezvous_key.clone(),
        );

        let (actual_sender, value) = cell.take().await;
        match actual_sender {
            Some(actual_sender) => {
                if *sender != actual_sender {
                    Err(Error::Networking(format!(
                        "wrong sender; expected {:?} but got {:?}",
                        sender, actual_sender
                    )))
                } else {
                    #[cfg(debug_assertions)]
                    tracing::debug!("Received '{}' from {}", rendezvous_key, sender);
                    Ok(value)
                }
            }
            None => {
                #[cfg(debug_assertions)]
                tracing::debug!("Received '{}' from {}", rendezvous_key, sender);
                Ok(value)
            }
        }
    }
}

impl Drop for GrpcNetworking {
    fn drop(&mut self) {
        let _ = self.stores.remove(&self.session_id);
    }
}

type AuthValue = (Option<Identity>, Value);

type SessionStore = DashMap<RendezvousKey, Arc<AsyncCell<AuthValue>>>;
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
) -> Arc<AsyncCell<AuthValue>> {
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
        let sender = crate::grpc::extract_sender(&request)
            .map_err(|e| tonic::Status::new(tonic::Code::Aborted, e))?
            .map(Identity::from);

        let request = request.into_inner();
        let tagged_value =
            bincode::deserialize::<TaggedValue>(&request.tagged_value).map_err(|_e| {
                tonic::Status::new(tonic::Code::Aborted, "failed to parse value".to_string())
            })?;

        let cell = cell(
            &self.stores,
            tagged_value.session_id,
            tagged_value.rendezvous_key,
        );
        cell.set((sender, tagged_value.value));

        Ok(tonic::Response::new(SendValueResponse::default()))
    }
}

#[derive(Serialize, Deserialize)]
struct TaggedValue {
    session_id: SessionId,
    rendezvous_key: RendezvousKey,
    value: Value,
}
