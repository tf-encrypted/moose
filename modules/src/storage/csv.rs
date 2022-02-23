use moose::error::Error;
use moose::prelude::*;
use moose::Result;
use ndarray::{Array2, ShapeBuilder};
use std::collections::{HashMap, HashSet};

pub async fn read_csv(
    filename: &str,
    _maybe_type_hint: Option<Ty>,
    columns: &[String],
    placement: &str,
) -> Result<Value> {
    let mut include_columns: HashSet<String> = HashSet::new();
    for column_name in columns.iter() {
        include_columns.insert(column_name.clone());
    }
    let mut reader = csv::Reader::from_path(filename)
        .map_err(|e| Error::Storage(format!("could not open file: {}: {}", filename, e)))?;
    let mut data: HashMap<String, Vec<f64>> = HashMap::new();
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| Error::Storage(format!("could not get headers from: {}: {}", filename, e)))?
        .into_iter()
        .map(|header| header.to_string())
        .collect();
    for header in &headers {
        data.insert(header.clone(), Vec::new());
    }

    for result in reader.records() {
        let record = result.map_err(|e| {
            Error::Storage(format!("could not get record from: {}: {}", filename, e))
        })?;
        for (header, value) in headers.iter().zip(record.iter()) {
            if include_columns.contains(header) || include_columns.is_empty() {
                data.entry(header.to_string())
                    .or_default()
                    .push(value.parse::<f64>().map_err(|e| {
                        Error::Storage(format!("could not parse '{}' to f64: {}", value, e))
                    })?);
            }
        }
    }

    let ncols = data.len();
    let nrows = data[&headers[0]].len();
    let shape = (nrows, ncols).f();
    let mut matrix: Vec<f64> = Vec::new();
    for header in headers {
        let column = &data[&header];
        matrix.extend_from_slice(column);
    }
    let ndarr: Array2<f64> = Array2::from_shape_vec(shape, matrix).map_err(|e| {
        Error::Storage(format!(
            "could not convert data from: {} to matrix: {}",
            filename, e
        ))
    })?;
    let plc = HostPlacement::from(placement);
    let tensor: HostFloat64Tensor = plc.from_raw(ndarr);
    Ok(Value::from(tensor))
}
