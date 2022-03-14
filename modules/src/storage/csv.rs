use moose::error::Error;
use moose::prelude::*;
use moose::Result;
use ndarray::{Array2, ShapeBuilder};
use std::collections::{HashMap, HashSet};

pub async fn read_csv(
    filename: &str,
    columns: &[String],
    placement: &HostPlacement,
) -> Result<Value> {
    let include_columns: HashSet<&String> = columns.iter().collect();
    let mut reader = csv::Reader::from_path(filename)
        .map_err(|e| Error::Storage(format!("could not open file: {}: {}", filename, e)))?;
    let mut data: HashMap<String, Vec<f64>> = HashMap::new();
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| Error::Storage(format!("could not get headers from: {}: {}", filename, e)))?
        .into_iter()
        .map(|header| header.to_string())
        .collect();
    if headers.is_empty() {
        return Err(Error::Storage(format!(
            "no columns found for file: {}",
            filename
        )));
    }
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
    let tensor: HostFloat64Tensor = placement.from_raw(ndarr);
    Ok(Value::from(tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_read_csv() {
        let plc = HostPlacement::from("host");
        let arr = array![[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]];
        let tensor: HostFloat64Tensor = plc.from_raw(arr);
        let expected = Value::from(tensor);
        let file_data = concat!("col_0,col_1\n", "1.1,2.2\n", "3.3,4.4\n", "5.5,6.6\n");
        let mut file = NamedTempFile::new().expect("trying to create tempfile");
        file.write_all(file_data.as_bytes()).unwrap();
        let path = file.path();
        let filename = path
            .to_str()
            .expect("trying to get path from temp file")
            .to_string();

        let plc = HostPlacement::from("host");
        let data = read_csv(&filename, &[], &plc).await.unwrap();
        assert_eq!(data, expected);
    }
}
