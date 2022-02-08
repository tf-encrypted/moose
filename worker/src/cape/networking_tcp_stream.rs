use async_trait::async_trait;
use moose::{
    computation::{RendezvousKey, SessionId, Value},
    execution::Identity,
    networking::AsyncNetworking,
};

#[derive(Default, Debug)]
pub struct TcpStreamNetworking {}

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
