use async_trait::async_trait;
use moose::{
    computation::{RendezvousKey, SessionId, Value},
    execution::Identity,
    networking::AsyncNetworking,
};
use std::collections::HashMap;
use std::io::prelude::*;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

type StoreType = Arc<dashmap::DashMap<String, Arc<async_cell::sync::AsyncCell<Value>>>>;

pub struct TcpStreamNetworking {
    own_name: String,
    hosts: HashMap<String, String>,
    store: StoreType,
    streams: HashMap<String, TcpStream>,
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

fn handle_connection(mut stream: TcpStream, store: StoreType) -> anyhow::Result<()> {
    loop {
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
        let value: Value = bincode::deserialize(&vec)
            .map_err(|e| anyhow::anyhow!("failed to deserialize moose value: {}", e))?;
        println!("got moose value: {:?}", value);

        // put value into store
        let rendezvous_key = "1234".to_string(); // TODO: get rendezvous_key via protocol
        let cell = store
            .entry(rendezvous_key)
            .or_insert_with(async_cell::sync::AsyncCell::shared)
            .value()
            .clone();

        cell.set(value);
    }
}

fn server(listener: TcpListener, store: StoreType) -> anyhow::Result<()> {
    loop {
        let (stream, _addr) = listener.accept().unwrap();
        let shared_store = Arc::clone(&store);
        tokio::spawn(async move {
            handle_connection(stream, shared_store).unwrap();
        });
    }
}

impl TcpStreamNetworking {
    pub async fn new(
        own_name: &str,
        hosts: HashMap<String, String>,
    ) -> anyhow::Result<TcpStreamNetworking> {
        let store = StoreType::default();
        let own_name: String = own_name.to_string();
        let mut streams = HashMap::new();
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
                streams.insert(placement.clone(), stream);
                break;
            }
        }

        let store = Arc::clone(&store);
        Ok(TcpStreamNetworking {
            own_name,
            hosts,
            store,
            streams,
        })
    }
}

#[async_trait]
impl AsyncNetworking for TcpStreamNetworking {
    async fn send(
        &self,
        _value: &Value,
        _receiver: &Identity,
        _rendezvous_key: &RendezvousKey,
        _session_id: &SessionId,
    ) -> moose::error::Result<()> {
        unimplemented!("network stub")
    }

    async fn receive(
        &self,
        _sender: &Identity,
        _rendezvous_key: &RendezvousKey,
        _session_id: &SessionId,
    ) -> moose::error::Result<Value> {
        unimplemented!("network stub")
    }
}
