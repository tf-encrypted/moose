use moose::prelude::*;
use moose::Result;
use ndarray::ArrayD;
use ndarray_npy::read_npy;
use std::io::Read;

pub async fn read_numpy(filename: &str, placement: &HostPlacement) -> Result<Value> {
    let arr: ArrayD<f64> = read_npy(filename).unwrap();
    let tensor: HostFloat64Tensor = placement.from_raw(arr);
    let value = Value::from(tensor);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_read_numpy() {
        let plc = HostPlacement::from("host");
        let tensor: HostFloat64Tensor = plc.from_raw(array![
            [[2.3, 4.0, 5.0], [6.0, 7.0, 12.0]],
            [[8.0, 9.0, 14.0], [10.0, 11.0, 16.0]]
        ]);
        let expected = Value::from(tensor);
        let mut file = NamedTempFile::new().expect("trying to create tempfile");
        let path = file.path();
        let filename = path
            .to_str()
            .expect("trying to get path from temp file")
            .to_string();

        let file_data = concat!(
            "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBl",
            "JzogKDIsIDIsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg",
            "ICAgICAgICAgICAgIApmZmZmZmYCQAAAAAAAABBAAAAAAAAAFEAAAAAAAAAYQAAAAAAAABxAAAAA",
            "AAAAKEAAAAAAAAAgQAAAAAAAACJAAAAAAAAALEAAAAAAAAAkQAAAAAAAACZAAAAAAAAAMEA="
        );
        let raw_bytes = base64::decode(file_data).unwrap();
        file.write_all(&raw_bytes).unwrap();

        let plc = HostPlacement::from("host");
        let data = read_numpy(&filename, &plc).await.unwrap();
        assert_eq!(data, expected);
    }
}
