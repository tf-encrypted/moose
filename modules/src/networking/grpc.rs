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

    pub fn new_session(
        &self,
        session_id: SessionId,
        tls_config: Option<ClientTlsConfig>,
    ) -> Arc<impl AsyncNetworking> {
        Arc::new(GrpcNetworking {
            session_id,
            stores: Arc::clone(&self.stores),
            channels: Arc::clone(&self.channels),
            tls_config,
        })
    }
}

struct GrpcNetworking {
    tls_config: Option<ClientTlsConfig>,
    session_id: SessionId,
    stores: Arc<SessionStores>,
    channels: Arc<Channels>,
}

use tonic::transport::ClientTlsConfig;
impl GrpcNetworking {
    fn channel(&self, receiver: &Identity) -> moose::Result<Channel> {
        tracing::debug!("Identity: {:?}", receiver);
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

                let channel = Channel::builder(endpoint);
                let channel = match self.tls_config.clone() {
                    Some(tls_config) => channel.tls_config(tls_config).map_err(|e| {
                        moose::Error::Networking(format!(
                            "failed to TLS config {:?}",
                            e.to_string()
                        ))
                    })?,
                    None => channel,
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
                if sender != &actual_sender {
                    Err(moose::Error::Networking(format!(
                        "wrong CA validation. Expected {:?} but got {:?}",
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
        use openssl::nid::Nid;
        use openssl::x509::X509;
        use tonic::{Code, Status};

        let certs = request.peer_certs();

        let request = request.into_inner();
        let tagged_value = bincode::deserialize::<TaggedValue>(&request.tagged_value).unwrap(); // TODO error handling

        match certs {
            Some(certs) => {
                if certs.len() != 1 {
                    tracing::debug!("Returning Aborted status since SSL cert has a large chain");
                    Status::new(
                        Code::Aborted,
                        format!(
                            "Cannot extract identity from an SSL Cert chain of length {:?}",
                            certs.len()
                        ),
                    );
                } else {
                    let cert = X509::from_der(certs[0].get_ref()).map_err(|err| {
                        Status::new(
                            Code::Aborted,
                            format!(
                                "Error parsing the X509 certificate with err: {:?}",
                                err.to_string()
                            ),
                        )
                    })?;
                    let subject = cert.subject_name();
                    let cn = subject.entries_by_nid(Nid::COMMONNAME).next().unwrap();

                    let cn_string = std::str::from_utf8(cn.data().as_slice())
                        .map_err(|err| Status::new(Code::Aborted, err.to_string()))?;
                    let sender = Identity::from(cn_string);
                    let cell = cell(
                        &self.stores,
                        tagged_value.session_id,
                        tagged_value.rendezvous_key,
                    );
                    cell.set((Some(sender), tagged_value.value));
                }
            }
            None => {
                let cell = cell(
                    &self.stores,
                    tagged_value.session_id,
                    tagged_value.rendezvous_key,
                );
                cell.set((None, tagged_value.value));
            }
        }
        Ok(tonic::Response::new(SendValueResponse::default()))
    }
}

#[derive(Serialize, Deserialize)]
struct TaggedValue {
    session_id: SessionId,
    rendezvous_key: RendezvousKey,
    value: Value,
}
