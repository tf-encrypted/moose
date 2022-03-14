use moose::error::Error;
use moose::prelude::*;
use moose::Result;
use ndarray::ArrayD;
use ndarray_npy::read_npy;
use std::fs::File;
use std::io::Read;

enum NumpyDtype {
    Float32,
    Float64,
    Int32,
    Int64,
    Uint32,
    Uint64,
}

#[allow(dead_code)]
pub(crate) async fn read_numpy(filename: &str, placement: &HostPlacement) -> Result<Value> {
    let dtype = extract_dtype(filename)?;
    match dtype {
        NumpyDtype::Float64 => {
            let arr: ArrayD<_> = read_npy(filename).map_err(|e| {
                Error::Storage(format!(
                    "failed to read numpy data file: {}: {}",
                    filename, e
                ))
            })?;
            let tensor: HostFloat64Tensor = placement.from_raw(arr);
            let value = Value::from(tensor);
            Ok(value)
        }
        NumpyDtype::Float32 => {
            let arr: ArrayD<_> = read_npy(filename).map_err(|e| {
                Error::Storage(format!(
                    "failed to read numpy data file: {}: {}",
                    filename, e
                ))
            })?;
            let tensor: HostFloat32Tensor = placement.from_raw(arr);
            let value = Value::from(tensor);
            Ok(value)
        }
        NumpyDtype::Int32 => {
            let arr: ArrayD<_> = read_npy(filename).map_err(|e| {
                Error::Storage(format!(
                    "failed to read numpy data file: {}: {}",
                    filename, e
                ))
            })?;
            let tensor: HostInt32Tensor = placement.from_raw(arr);
            let value = Value::from(tensor);
            Ok(value)
        }
        NumpyDtype::Int64 => {
            let arr: ArrayD<_> = read_npy(filename).map_err(|e| {
                Error::Storage(format!(
                    "failed to read numpy data file: {}: {}",
                    filename, e
                ))
            })?;
            let tensor: HostInt64Tensor = placement.from_raw(arr);
            let value = Value::from(tensor);
            Ok(value)
        }
        NumpyDtype::Uint64 => {
            let arr: ArrayD<_> = read_npy(filename).map_err(|e| {
                Error::Storage(format!(
                    "failed to read numpy data file: {}: {}",
                    filename, e
                ))
            })?;
            let tensor: HostUint64Tensor = placement.from_raw(arr);
            let value = Value::from(tensor);
            Ok(value)
        }
        NumpyDtype::Uint32 => {
            let arr: ArrayD<_> = read_npy(filename).map_err(|e| {
                Error::Storage(format!(
                    "failed to read numpy data file: {}: {}",
                    filename, e
                ))
            })?;
            let tensor: HostUint32Tensor = placement.from_raw(arr);
            let value = Value::from(tensor);
            Ok(value)
        }
    }
}

fn match_char(got: u8, expected: char) -> Result<()> {
    if got != expected as u8 {
        Err(Error::Storage(format!(
            "expecting: {} got: {}",
            expected, got
        )))
    } else {
        Ok(())
    }
}

fn consume_spaces(file: &mut File) -> Result<u8> {
    loop {
        let c = getc(file)?;
        if c != b' ' {
            return Ok(c);
        }
    }
}

fn getc(file: &mut File) -> Result<u8> {
    let mut buf: [u8; 1] = [0; 1];
    file.read(&mut buf)
        .map_err(|e| Error::Storage(format!("failed to read byte from file: {}", e)))?;
    let byte = buf[0];
    Ok(byte)
}

// Lexical analysis of the numpy data file to find the dtype
// description of numpy binary file format here:
//     https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
fn extract_descr(file: &mut File) -> Result<Vec<char>> {
    // First 10 bytes are magic numbers
    for _ in 0..10 {
        getc(file)?;
    }
    let c = getc(file)?;

    // Found start of dictionary
    match_char(c, '{')?;
    let c = consume_spaces(file)?;
    match_char(c, '\'')?;

    // Find the key "descr". This is the entry for the dtype of the numpy object
    loop {
        let mut word: String = String::new();
        loop {
            let c = getc(file)?;
            if c == b'\'' || c == b'"' {
                break;
            }
            word.push(c as char);
        }
        if word == "descr" {
            break;
        }
    }
    match_char(c, '\'')?;

    let c = consume_spaces(file)?;

    // ':' denotes the beginning of the value section for this dict entry
    match_char(c, ':')?;
    let c = consume_spaces(file)?;
    match_char(c, '\'')?;

    // Now we are at the value corresponding to the "descr" key in the
    // dictionary. Let's now read what the value actually is.
    let mut descr = Vec::new();
    loop {
        let c = getc(file)?;
        if c == b'\'' {
            break;
        }
        descr.push(c as char);
    }

    if descr.is_empty() {
        Err(Error::Storage(
            "could not find \"descr\" in numpy data dictionary".to_string(),
        ))
    } else {
        Ok(descr)
    }
}

fn descr_to_dtype(descr: &[char]) -> Result<NumpyDtype> {
    if descr.is_empty() {
        return Err(Error::Storage(
            "descr is empty in numpy data dictionary".to_string(),
        ));
    }

    // we can ignore byte order marks to get the dtype
    let dtype_start = if descr[0] == '<' || descr[0] == '>' {
        1
    } else {
        0
    };

    let letter_code = descr
        .get(dtype_start)
        .ok_or_else(|| Error::Storage("missing letter code from numpy file descr".to_string()))?;
    let number_code = descr.get(dtype_start + 1);

    // letter_code:
    //     specifies overall type, e.g., float is f, int is i, uint is u.
    // number_code:
    //     specifies the number of bytes, e.g., 4 means 32 bits, 8 means 64 bits
    match (letter_code, number_code) {
        ('f', Some('4')) => Ok(NumpyDtype::Float32),
        ('f', Some('8')) => Ok(NumpyDtype::Float64),
        ('d', None) => Ok(NumpyDtype::Float64),
        ('i', Some('4')) => Ok(NumpyDtype::Int32),
        ('i', Some('8')) => Ok(NumpyDtype::Int64),
        ('u', Some('4')) => Ok(NumpyDtype::Uint32),
        ('u', Some('8')) => Ok(NumpyDtype::Uint64),
        _ => {
            let number_code_display = match number_code {
                Some(c) => c.to_string(),
                None => String::new(),
            };
            Err(Error::Storage(format!(
                "unknown numpy descr: {}{}",
                letter_code, number_code_display
            )))
        }
    }
}

fn extract_dtype(npy_filename: &str) -> Result<NumpyDtype> {
    let mut file = std::fs::File::open(npy_filename).map_err(|e| {
        Error::Storage(format!(
            "failed to open numpy data file for reading: {}: {}",
            npy_filename, e
        ))
    })?;
    let descr = extract_descr(&mut file)?;
    let numpy_dtype = descr_to_dtype(&descr)?;
    Ok(numpy_dtype)
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
