use async_trait::async_trait;
use moose::computation::SessionId;
use moose::computation::Ty;
use moose::computation::Value;
use moose::error::Result;
use moose::storage::AsyncStorage;

pub struct StubAsyncStorage {}

impl StubAsyncStorage {
    #[allow(dead_code)]
    pub fn new(_placement: &str) -> StubAsyncStorage {
        StubAsyncStorage {}
    }
}

impl Default for StubAsyncStorage {
    fn default() -> Self {
        Self::new("host")
    }
}

#[async_trait]
impl AsyncStorage for StubAsyncStorage {
    async fn save(&self, _key: &str, _session_id: &SessionId, _val: &Value) -> Result<()> {
        unimplemented!("this is a non-functioning stub storage")
    }

    async fn load(
        &self,
        _key: &str,
        _session_id: &SessionId,
        _type_hint: Option<Ty>,
        _query: &str,
    ) -> Result<Value> {
        unimplemented!("this is a non-functioning stub storage")
    }
}
