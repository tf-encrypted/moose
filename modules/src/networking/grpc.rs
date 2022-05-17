mod gen {
    tonic::include_proto!("moose_networking");
}

use self::gen::networking_client::NetworkingClient;
use self::gen::networking_server::{Networking, NetworkingServer};
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
use tonic::transport::{Channel, ClientTlsConfig, Uri};
use x509_parser::prelude::*;

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

struct GrpcNetworking {
    tls_config: Option<ClientTlsConfig>,
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

                let mut channel = Channel::builder(endpoint);
                if let Some(ref tls_config) = self.tls_config {
                    channel = channel.tls_config(tls_config.clone()).map_err(|e| {
                        moose::Error::Networking(format!(
                            "failed to TLS config {:?}",
                            e.to_string()
                        ))
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
        sender: &Identity,
        rendezvous_key: &RendezvousKey,
        _session_id: &SessionId,
    ) -> moose::Result<Value> {
        let cell = cell(
            &self.stores,
            self.session_id.clone(),
            rendezvous_key.clone(),
        );

        let (actual_sender, value) = cell.take().await;
        match actual_sender {
            Some(actual_sender) => {
                if *sender != actual_sender {
                    Err(moose::Error::Networking(format!(
                        "wrong sender; expected {:?} but got {:?}",
                        sender, actual_sender
                    )))
                } else {
                    Ok(value)
                }
            }
            None => Ok(value),
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
        let sender =
            extract_sender(&request).map_err(|e| tonic::Status::new(tonic::Code::Aborted, e))?;

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

fn extract_sender(request: &tonic::Request<SendValueRequest>) -> Result<Option<Identity>, String> {
    match request.peer_certs() {
        None => Ok(None),
        Some(certs) => {
            if certs.len() != 1 {
                return Err(format!(
                    "cannot extract identity from certificate chain of length {:?}",
                    certs.len()
                ));
            }

            let (_rem, cert) = parse_x509_certificate(certs[0].as_ref()).map_err(|err| {
                format!("failed to parse X509 certificate: {:?}", err.to_string())
            })?;

            let cns: Vec<_> = cert
                .subject()
                .iter_common_name()
                .map(|attr| attr.as_str().map_err(|err| err.to_string()))
                .collect::<Result<_, _>>()?;

            if let Some(cn) = cns.first() {
                let sender = Identity::from(*cn);
                Ok(Some(sender))
            } else {
                Err("certificate common name was empty".to_string())
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct TaggedValue {
    session_id: SessionId,
    rendezvous_key: RendezvousKey,
    value: Value,
}
