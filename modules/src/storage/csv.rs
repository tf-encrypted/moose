use moose::prelude::*;
use ndarray::{Array2, ShapeBuilder};
use std::collections::{HashMap, HashSet};

pub async fn read_csv(
    filename: &str,
    _maybe_type_hint: Option<Ty>,
    columns: &[String],
    placement: &str,
) -> anyhow::Result<Value> {
    let mut include_columns: HashSet<String> = HashSet::new();
    for column_name in columns.iter() {
        include_columns.insert(column_name.clone());
    }
    let mut reader = csv::Reader::from_path(filename)?;
    let mut data: HashMap<String, Vec<f64>> = HashMap::new();
    let headers: Vec<String> = reader
        .headers()?
        .into_iter()
        .map(|header| header.to_string())
        .collect();
    for header in &headers {
        data.insert(header.clone(), Vec::new());
    }

    for result in reader.records() {
        let record = result?;
        for (header, value) in headers.iter().zip(record.iter()) {
            if include_columns.contains(header) || include_columns.is_empty() {
                data.entry(header.to_string())
                    .or_default()
                    .push(value.parse::<f64>()?);
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
    let ndarr: Array2<f64> = Array2::from_shape_vec(shape, matrix)?;
    let plc = HostPlacement::from(placement);
    let tensor: HostFloat64Tensor = plc.from_raw(ndarr);
    Ok(Value::from(tensor))
}
