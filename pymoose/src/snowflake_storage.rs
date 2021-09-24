use async_trait::async_trait;
use moose::computation::{SessionId, Ty, Value};
use moose::error::Result;
use moose::host::HostFloat64Tensor;
use moose::storage::AsyncStorage;
use numpy::PyArrayDyn;
use pyo3::prelude::*;

pub struct SnowflakeStorage {
    user: String,
    password: String,
    account_identifier: String,
    aes_key: String,
}

impl SnowflakeStorage {
    pub fn new() -> Self {
        let user = std::env::var("CAPE_SNOWFLAKE_USER").unwrap();
        let password = std::env::var("CAPE_SNOWFLAKE_PASSWORD").unwrap();
        let account_identifier = std::env::var("CAPE_SNOWFLAKE_ACCOUNT_IDENTIFIER").unwrap();
        let aes_key = std::env::var("CAPE_SNOWFLAKE_AES_KEY").unwrap();
        SnowflakeStorage {
            user,
            password,
            account_identifier,
            aes_key,
        }
    }
}

fn generate_python_names() -> (String, String) {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    const STRING_LEN: usize = 30;

    use rand::Rng;
    let mut rng = rand::thread_rng();

    let file_name: String = (0..STRING_LEN)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect();
    let module_name = file_name.clone();

    (file_name + ".py", module_name)
}

#[async_trait]
impl AsyncStorage for SnowflakeStorage {
    async fn save(&self, _key: &str, _session_id: &SessionId, _val: &Value) -> Result<()> {
        unimplemented!()
    }

    async fn load(
        &self,
        key: &str,
        _session_id: &SessionId,
        _type_hint: Option<Ty>,
        query: &str,
    ) -> Result<Value> {
        pyo3::prepare_freethreaded_python();
        let gil = Python::acquire_gil();
        let py = gil.python();

        // TODO maybe this could be done once and for all instead of on each load()
        let (file_name, module_name) = generate_python_names();
        let python_module = PyModule::from_code(py, LOAD_SCRIPT, &file_name, &module_name)
            .map_err(|e| {
                e.print(py);
                moose::error::Error::Storage(format!(
                    "Error generating Python module for loading data"
                ))
            })?;

        let py_res = python_module
            .getattr("load_data")
            .map_err(|e| {
                e.print(py);
                moose::error::Error::Storage(format!("Python load script is malformed"))
            })?
            .call1((
                self.user.to_object(py),
                self.password.to_object(py),
                self.account_identifier.to_object(py),
                self.aes_key.to_object(py),
                key.to_object(py),
                query.to_object(py),
            ))
            .map_err(|e| {
                e.print(py);
                moose::error::Error::Storage(format!("Failed to load data in Python"))
            })?
            .to_object(py);

        let raw_data = py_res
            .cast_as::<PyArrayDyn<f64>>(py)
            .map_err(|e| {
                moose::error::Error::Storage(format!(
                    "Failed to convert data from Python to Rust: {:?}",
                    e
                ))
            })?
            .to_owned_array();

        let val = HostFloat64Tensor::from(raw_data).into();
        Ok(val)
    }
}

const LOAD_SCRIPT: &str = r#"
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import sys
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES


class AESCipher(object):

    def __init__(self, key): 
        self.bs = AES.block_size
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw.encode()))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]


def load_data(user, password, account_identifier, aes_key, key, query):

    engine = create_engine(
        'snowflake://{user}:{password}@{account_identifier}/'.format(
            user=user,
            password=password,
            account_identifier=account_identifier,
        )
    )

    try:
        connection = engine.connect()
        pd.read_sql_query("use CAPE.PUBLIC", engine)
        df = pd.read_sql_query(query, engine)

    finally:
        connection.close()
        engine.dispose()

    series = df[df.columns[0]]

    aes = AESCipher(aes_key)
    for i, value in enumerate(series):
        series[i] = float(aes.decrypt(value))

    res = series.to_numpy().astype(np.float64)
    return res
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_snowflake() {
        let storage = SnowflakeStorage::new();
        let val = storage
            .load(
                "my_key",
                &SessionId::random(),
                None,
                "SELECT total_sales FROM sales",
            )
            .await
            .unwrap();

        println!("{:?}", val);
    }
}
