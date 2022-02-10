use async_trait::async_trait;
use moose::{
    computation::{RendezvousKey, SessionId, Value},
    execution::Identity,
    networking::AsyncNetworking,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::prelude::*;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

type StoreType = Arc<dashmap::DashMap<String, Arc<async_cell::sync::AsyncCell<Value>>>>;

pub struct TcpStreamNetworking {
    own_name: String,                                       // name of own placement
    hosts: HashMap<String, String>,                         // host to address mapping
    store: StoreType,                                       // store incoming data
    send_channels: HashMap<String, mpsc::Sender<SendData>>, // send data over each stream
}

fn u64_to_little_endian(n: u64, buf: &mut [u8; 8]) -> anyhow::Result<()> {
    let mut n_mut = n;
    for i in 0..=7 {
        buf[i] = (n_mut & 0xff) as u8;
        n_mut >>= 8;
    }
    Ok(())
}

fn little_endian_to_u64(buf: &[u8; 8]) -> u64 {
    let mut n: u64 = 0;
    for i in 0..=7 {
        n |= (buf[i] as u64) << (i * 8);
    }
    n
}

fn compute_path(session_id: &SessionId, rendezvous_key: &RendezvousKey) -> String {
    format!("{}/{}", session_id, rendezvous_key)
}

fn send_value(mut stream: &TcpStream, send_data: &SendData) -> anyhow::Result<()> {
    let raw_data: Vec<u8> = bincode::serialize(send_data)?;
    let data_size = raw_data.len();
    let mut size_data_buf = [0; 8];
    u64_to_little_endian(data_size.try_into()?, &mut size_data_buf)?;

    // TODO: write error, re-establish connection?
    stream.write_all(&size_data_buf)?;
    stream.write_all(&raw_data)?;
    Ok(())
}

fn handle_connection(mut stream: TcpStream, store: StoreType) -> anyhow::Result<()> {
    loop {
        // read moose data
        let mut buf: [u8; 8] = [0; 8];
        let size = match stream.read_exact(&mut buf) {
            Ok(_) => little_endian_to_u64(&buf),
            Err(_) => return Ok(()), // when client hangs up
        };
        let mut vec: Vec<u8> = Vec::with_capacity(size as usize);
        unsafe {
            // https://stackoverflow.com/a/28209155
            vec.set_len(size as usize);
        }

        stream.read_exact(&mut vec)?;
        let data: SendData = bincode::deserialize(&vec)
            .map_err(|e| anyhow::anyhow!("failed to deserialize moose value: {}", e))?;
        //println!("got moose value: {:?}", data);

        // put value into store
        let key = compute_path(&data.session_id, &data.rendezvous_key);
        let cell = store
            .entry(key)
            .or_insert_with(async_cell::sync::AsyncCell::shared)
            .value()
            .clone();

        cell.set(data.value);
    }
}

fn server(listener: TcpListener, store: StoreType) -> anyhow::Result<()> {
    loop {
        // TODO: retry logic?
        let (stream, _addr) = listener.accept().unwrap();
        let shared_store = Arc::clone(&store);
        tokio::spawn(async move {
            handle_connection(stream, shared_store).unwrap();
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

async fn send_loop(stream: TcpStream, mut rx: mpsc::Receiver<SendData>) -> anyhow::Result<()> {
    loop {
        let send_data = rx.recv().await;
        match send_data {
            Some(data) => {
                send_value(&stream, &data)?;
            }
            None => {
                unimplemented!("I think we should shutdown the computation now?")
            }
        }
    }
}

impl TcpStreamNetworking {
    pub async fn new(
        own_name: &str,
        hosts: HashMap<String, String>,
    ) -> anyhow::Result<TcpStreamNetworking> {
        let store = StoreType::default();
        let own_name: String = own_name.to_string();
        let own_address = hosts
            .get(&own_name)
            .ok_or_else(|| anyhow::anyhow!("own host name not in hosts map"))?;

        // spawn the server
        println!("spawned server on: {}", own_address);
        let listener = TcpListener::bind(&own_address)?;
        let shared_store = Arc::clone(&store);
        tokio::spawn(async move {
            server(listener, Arc::clone(&shared_store)).unwrap();
        });

        // connect to every other server
        let mut others: Vec<(String, String)> = hosts
            .clone()
            .into_iter()
            .filter(|(placement, _)| *placement != own_name)
            .collect();
        others.sort();
        println!("others = {:?}", others);
        let mut send_channels = HashMap::new();
        for (placement, address) in others.iter() {
            println!("trying: {} -> {}", placement, address);
            loop {
                let stream = match TcpStream::connect(address) {
                    Ok(s) => s,
                    Err(_) => {
                        sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                };
                println!("connected to: {} -> {}", placement, address);
                let (tx, rx) = mpsc::channel(100);
                send_channels.insert(placement.clone(), tx);

                tokio::spawn(async move {
                    send_loop(stream, rx).await.unwrap();
                });
                break;
            }
        }

        let store = Arc::clone(&store);
        Ok(TcpStreamNetworking {
            own_name,
            hosts,
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
    ) -> moose::error::Result<()> {
        let key = compute_path(session_id, rendezvous_key);
        println!("sent key: {}", key);
        let receiver_name = receiver.to_string();
        let send_channel = self.send_channels.get(&receiver_name).unwrap();
        let send_data = SendData {
            value: value.clone(),
            receiver: receiver.clone(),
            rendezvous_key: rendezvous_key.clone(),
            session_id: session_id.clone(),
        };
        println!("awaiting send: {}", key);
        send_channel.send(send_data).await;

        Ok(())
    }

    async fn receive(
        &self,
        _sender: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> moose::error::Result<Value> {
        let key = compute_path(&session_id, &rendezvous_key);
        println!("got key: {}", key);

        let cell = self
            .store
            .entry(key.clone())
            .or_insert_with(async_cell::sync::AsyncCell::shared)
            .value()
            .clone();

        println!("awaiting receive key: {}", key);
        let value = cell.get().await;
        // TODO: delete entry from dashmap?
        Ok(value)
    }
}
