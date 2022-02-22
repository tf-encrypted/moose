use moose::prelude::*;
use ndarray::{Array2, ShapeBuilder};
use npy::NpyData;
use std::io::Read;

pub async fn read_numpy(
    filename: &str,
    _maybe_type_hint: Option<Ty>,
    _columns: &[String],
    _placement: &str,
) -> anyhow::Result<Value> {
    let mut buf = vec![];
    std::fs::File::open(filename)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();

    let data: NpyData<f64> = NpyData::from_bytes(&buf).unwrap();
    for number in data {
        eprintln!("{}", number);
    }
    unimplemented!("numpy");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_read_numpy() {
        let path = std::env::current_dir().unwrap();
        println!("The current directory is {}", path.display());
        read_numpy("data.npy", None, &[], "host").await.unwrap();
    }
}
