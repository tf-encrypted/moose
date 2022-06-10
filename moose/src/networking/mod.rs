//! Networking traits and implementations.

use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::Identity;
use async_trait::async_trait;

mod constants;
pub mod grpc;
pub mod local;
pub mod tcpstream;

/// Requirements for synchronous networking.
///
/// An implementation of this trait must be provided when using Moose
/// for synchronous (blocking) execution.
pub trait SyncNetworking {
    fn send(
        &self,
        value: &Value,
        receiver: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()>;

    fn receive(
        &self,
        sender: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value>;
}

/// Requirements for asynchronous networking.
///
/// An implementation of this trait must be provided when using Moose
/// for asynchronous (blocking) execution.
#[async_trait]
pub trait AsyncNetworking {
    async fn send(
        &self,
        value: &Value,
        receiver: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<()>;

    async fn receive(
        &self,
        sender: &Identity,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value>;
}
