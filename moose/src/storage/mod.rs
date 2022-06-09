//! Storage traits and implementations.
//!
//! See also the `moose_modules` crate for more implementations.

use crate::computation::*;
use crate::error::{Error, Result};
use async_trait::async_trait;

pub mod filesystem;
pub mod local;

pub trait SyncStorage {
    fn save(&self, key: &str, session_id: &SessionId, val: &Value) -> Result<()>;

    fn load(
        &self,
        key: &str,
        session_id: &SessionId,
        type_hint: Option<Ty>,
        query: &str,
    ) -> Result<Value>;
}

#[async_trait]
pub trait AsyncStorage {
    async fn save(&self, key: &str, session_id: &SessionId, val: &Value) -> Result<()>;

    async fn load(
        &self,
        key: &str,
        session_id: &SessionId,
        type_hint: Option<Ty>,
        query: &str,
    ) -> Result<Value>;
}
