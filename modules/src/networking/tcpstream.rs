use async_trait::async_trait;
use moose::tokio;
use moose::{
    computation::{RendezvousKey, SessionId, Value},
    error::Error,
    execution::Identity,
    networking::AsyncNetworking,
    Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

type StoreType =
    Arc<dashmap::DashMap<(SessionId, RendezvousKey), Arc<async_cell::sync::AsyncCell<Value>>>>;

type SendChannelsType = HashMap<Identity, mpsc::Sender<(SendData, mpsc::Sender<()>)>>;
pub struct TcpStreamNetworking {
    own_name: String,
    store: StoreType,                // store incoming data
    send_channels: SendChannelsType, // send data over each stream
}

fn u64_to_little_endian(n: u64, buf: &mut [u8; 8]) {
    let mut n_mut = n;
    for item in buf.iter_mut().take(8) {
        *item = (n_mut & 0xff) as u8;
        n_mut >>= 8;
    }
}

fn little_endian_to_u64(buf: &[u8; 8]) -> u64 {
    let mut n: u64 = 0;
    for (i, item) in buf.iter().enumerate().take(8) {
        n |= (*item as u64) << (i * 8);
    }
    n
}

async fn handle_connection(mut stream: TcpStream, store: StoreType) -> Result<()> {
    loop {
        // read moose data
        let mut buf: [u8; 8] = [0; 8];
        tracing::debug!("reading size buf");
        let size = match stream.read_exact(&mut buf).await {
            Ok(_) => little_endian_to_u64(&buf),
            Err(_) => {
                tracing::debug!("client hung up");
                return Ok(()); // when client hangs up
            }
        };
        let mut vec: Vec<u8> = vec![0; size as usize];

        tracing::debug!("reading exact: {}", size);
        stream.read_exact(&mut vec).await.map_err(|e| {
            Error::Networking(format!("failed to read data from TCP stream: {}", e))
        })?;
        let data: SendData = bincode::deserialize(&vec)
            .map_err(|e| Error::Networking(format!("failed to deserialize moose value: {}", e)))?;

        // put value into store
        let key = (data.session_id, data.rendezvous_key);
        tracing::debug!("storing key: {:?}", key);
        let cell = store
            .entry(key.clone())
            .or_insert_with(async_cell::sync::AsyncCell::shared)
            .value()
            .clone();

        cell.set(data.value);
        tracing::debug!("stored key: {:?}", key);
    }
}

async fn server(listener: TcpListener, store: StoreType) -> Result<()> {
    loop {
        // TODO: retry logic?
        tracing::debug!("listening");
        let (stream, addr) = listener
            .accept()
            .await
            .map_err(|e| Error::Networking(format!("failed to accept connection: {}", e)))?;
        tracing::debug!("accepted connection: {}", addr);
        let shared_store = Arc::clone(&store);
        tokio::spawn(async move {
            handle_connection(stream, shared_store).await.unwrap();
        });
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct SendData {
    value: Value,
    receiver: Identity,
    rendezvous_key: RendezvousKey,
    session_id: SessionId,
}

async fn send_value(stream: &mut TcpStream, send_data: &SendData) -> Result<()> {
    let raw_data: Vec<u8> = bincode::serialize(send_data)
        .map_err(|e| Error::Networking(format!("could not serialze send_data: {}", e)))?;
    let data_size = raw_data.len();
    let mut size_data_buf = [0; 8];
    u64_to_little_endian(
        data_size
            .try_into()
            .map_err(|e| Error::Networking(format!("could not cast data_size: {}", e)))?,
        &mut size_data_buf,
    );

    // TODO: write error, re-establish connection?
    stream.write_all(&size_data_buf).await.map_err(|e| {
        Error::Networking(format!("could not write data size over TCP stream: {}", e))
    })?;
    stream
        .write_all(&raw_data)
        .await
        .map_err(|e| Error::Networking(format!("could not write data over TCP stream: {}", e)))?;
    stream
        .flush()
        .await
        .map_err(|e| Error::Networking(format!("could not flush TCP stream: {}", e)))?;
    Ok(())
}

async fn send_loop(
    mut stream: TcpStream,
    mut rx: mpsc::Receiver<(SendData, mpsc::Sender<()>)>,
) -> Result<()> {
    loop {
        match rx.recv().await {
            Some((data, finished_send_signal)) => {
                send_value(&mut stream, &data).await.map_err(|e| {
                    Error::Networking(format!("could not send value to other worker: {}", e))
                })?;
                finished_send_signal.send(()).await.map_err(|e| {
                    Error::Networking(format!(
                        "could not transmit the finished sending ack over mpsc channel: {}",
                        e
                    ))
                })?;
            }
            None => {
                stream.shutdown().await.map_err(|e| {
                    Error::Networking(format!("failed to shutdown TCP stream: {}", e))
                })?;
                return Ok(());
            }
        }
    }
}

impl TcpStreamNetworking {
    pub async fn new(
        own_name: &str,
        hosts: HashMap<String, String>,
    ) -> Result<TcpStreamNetworking> {
        tracing::debug!("own name: {}", own_name);
        let store = StoreType::default();
        let own_name: String = own_name.to_string();
        let own_address = hosts
            .get(&own_name)
            .ok_or_else(|| Error::Networking("own host name not in hosts map".to_string()))?;

        // spawn the server
        tracing::debug!("spawned server on: {}", own_address);
        let listener = TcpListener::bind(&own_address).await.map_err(|e| {
            Error::Networking(format!("could not bind to address: {}: {}", own_address, e))
        })?;
        let shared_store = Arc::clone(&store);
        tokio::spawn(async move {
            server(listener, Arc::clone(&shared_store)).await.unwrap();
        });

        // connect to every other server
        let mut others: Vec<(String, String)> = hosts
            .clone()
            .into_iter()
            .filter(|(placement, _)| *placement != own_name)
            .collect();
        others.sort();
        tracing::debug!("others = {:?}", others);
        let mut send_channels = HashMap::new();
        for (placement, address) in others.iter() {
            tracing::debug!("trying: {} -> {}", placement, address);
            loop {
                let stream = match TcpStream::connect(address).await {
                    Ok(s) => s,
                    Err(_) => {
                        sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                };
                tracing::debug!("connected to: {} -> {}", placement, address);
                let (tx, rx) = mpsc::channel(100);
                let identity = Identity::from(placement);
                send_channels.insert(identity, tx);

                tokio::spawn(async move {
                    send_loop(stream, rx).await.unwrap();
                });
                break;
            }
        }

        let store = Arc::clone(&store);
        Ok(TcpStreamNetworking {
            own_name,
            store,
            send_channels,
        })
    }
}

#[async_trait]
impl AsyncNetworking for TcpStreamNetworking {
    async fn send(
        &self,
        value: &Value,
        receiver: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> moose::Result<()> {
        let key = (session_id, rendezvous_key);
        tracing::debug!("sending key: {:?} to: {}", key, receiver);
        let send_channel = self.send_channels.get(receiver).ok_or_else(|| {
            Error::Networking(format!(
                "in session {}, channel not found to send rendezvous key {} from {} to {}",
                session_id, rendezvous_key, self.own_name, receiver
            ))
        })?;
        let send_data = SendData {
            value: value.clone(),
            receiver: receiver.clone(),
            rendezvous_key: rendezvous_key.clone(),
            session_id: session_id.clone(),
        };
        tracing::debug!("awaiting send: {:?}", key);
        let (tx, mut send_finished) = mpsc::channel(1);
        send_channel.send((send_data, tx)).await.map_err(|e| {
            Error::Networking(format!(
                "in session {}, channel send failed for rendezvous key {} from {} to {}: {}",
                session_id, rendezvous_key, self.own_name, receiver, e
            ))
        })?;
        let ack = send_finished.recv().await.ok_or_else(|| {
            Error::Networking(
                "did not receive the finished sending ack from the send loop".to_string(),
            )
        })?;

        Ok(ack)
    }

    async fn receive(
        &self,
        sender: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> moose::Result<Value> {
        let key = (session_id.clone(), rendezvous_key.clone());
        tracing::debug!("receiving key: {:?} from: {}", key, sender);

        let cell = self
            .store
            .entry(key.clone())
            .or_insert_with(async_cell::sync::AsyncCell::shared)
            .value()
            .clone();

        tracing::debug!("awaiting receive key: {:?}", key);
        let value = cell.take().await;
        tracing::debug!("got key: {:?}", key);
        self.store.remove(&key).ok_or_else(|| {
            Error::Networking(format!("failed to remove key: {:?} from store", key))
        })?;
        Ok(value)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_u64_to_little_endian() {
        let x: u64 = 0xdeadbeefcafe1234;
        let mut buf: [u8; 8] = [0; 8];
        u64_to_little_endian(x, &mut buf);
        println!("{:x?}", buf);

        let test_int = little_endian_to_u64(&buf);
        println!("{:x?}", test_int);
        assert_eq!(test_int, x);
    }
}
