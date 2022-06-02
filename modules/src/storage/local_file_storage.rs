use crate::storage::csv::{read_csv, write_csv};
use crate::storage::numpy::{read_numpy, write_numpy};
use async_trait::async_trait;
use moose::error::Error;
use moose::prelude::*;
use moose::storage::AsyncStorage;
use moose::Result;
use std::path::Path;

#[derive(Default)]
struct AsyncLocalFileStorage {}

#[async_trait]
impl AsyncStorage for AsyncLocalFileStorage {
    async fn save(&self, key: &str, _session_id: &SessionId, val: &Value) -> Result<()> {
        let path = Path::new(key);
        let extension = path
            .extension()
            .ok_or_else(|| Error::Storage(format!("failed to get extension from key: {}", key)))?;
        match extension.to_str() {
            Some("csv") => write_csv(key, val).await,
            Some("npy") => write_numpy(key, val).await,
            _ => Err(Error::Storage(format!(
                "key must provide an extension of either '.csv' or '.npy', got: {}",
                key
            ))),
        }
    }

    async fn load(
        &self,
        key: &str,
        _session_id: &SessionId,
        type_hint: Option<Ty>,
        query: &str,
    ) -> Result<Value> {
        let path = Path::new(key);
        let extension = path
            .extension()
            .ok_or_else(|| Error::Storage(format!("failed to get extension from key: {}", key)))?;
        let plc = HostPlacement::from("host");
        match extension.to_str() {
            Some("csv") => {
                let query = parse_columns(query)?;
                read_csv(key, &query, &plc).await
            }
            Some("npy") => read_numpy(key, &plc, type_hint).await,
            _ => Err(Error::Storage(format!(
                "key must provide an extension of either '.csv' or '.npy', got: {}",
                key
            ))),
        }
    }
}

pub fn parse_columns(query: &str) -> Result<Vec<String>> {
    match query {
        "" => Ok(Vec::new()),
        query_str => {
            let jsn: serde_json::Value = serde_json::from_str(query_str)
                .map_err(|e| Error::Storage(format!("failed to parse query as json: {}", e)))?;
            let as_vec = match &jsn.get("select_columns") {
                Some(serde_json::Value::Array(v)) => v.to_vec(),
                _ => Vec::new(),
            };
            let select_columns: Result<Vec<String>> = as_vec
                .iter()
                .map(|i| match i {
                    serde_json::Value::String(s) => Ok(s.to_string()),
                    _ => Err(Error::Storage(
                        "select_columns must contain an array of strings of column names"
                            .to_string(),
                    )),
                })
                .collect();
            select_columns
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moose::tokio;
    use moose::types::HostFloat64Tensor;
    use ndarray::array;
    use std::convert::TryFrom;
    use std::fs::File;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_numpy_async_local_file_storage() {
        let storage = AsyncLocalFileStorage::default();

        let plc = HostPlacement::from("host");
        let tensor: HostFloat64Tensor = plc.from_raw(array![
            [[2.3, 4.0, 5.0], [6.0, 7.0, 12.0]],
            [[8.0, 9.0, 14.0], [10.0, 11.0, 16.0]]
        ]);
        let expected = Value::from(tensor);

        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("data.npy");

        let _file = File::create(&path).unwrap();
        let filename = path
            .to_str()
            .expect("trying to get path from temp file")
            .to_string();

        let session_id_str = "01FGSQ37YDJSVJXSA6SSY7G4Y2";
        let session_id = SessionId::try_from(session_id_str).unwrap();
        storage
            .save(&filename, &session_id, &expected)
            .await
            .unwrap();

        let data = storage
            .load(&filename, &session_id, None, "")
            .await
            .unwrap();
        assert_eq!(data, expected);
    }

    #[tokio::test]
    async fn test_csv_async_local_file_storage() {
        let storage = AsyncLocalFileStorage::default();

        let plc = HostPlacement::from("host");
        let tensor: HostFloat64Tensor = plc.from_raw(array![[2.3, 4.0, 5.0], [6.0, 7.0, 12.0]]);
        let expected = Value::from(tensor);

        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("data.csv");

        let _file = File::create(&path).unwrap();
        let filename = path
            .to_str()
            .expect("trying to get path from temp file")
            .to_string();

        let session_id_str = "01FGSQ37YDJSVJXSA6SSY7G4Y2";
        let session_id = SessionId::try_from(session_id_str).unwrap();
        storage
            .save(&filename, &session_id, &expected)
            .await
            .unwrap();

        let data = storage
            .load(&filename, &session_id, None, "")
            .await
            .unwrap();
        assert_eq!(data, expected);
    }
}
