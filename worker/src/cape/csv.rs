use arrow::record_batch::RecordBatch;
use datafusion::prelude::{CsvReadOptions, ExecutionContext};
use moose::error::Error;
use moose::prelude::*;
use ndarray::{Array2, ShapeBuilder};

pub async fn read_csv(
    filename: &str,
    maybe_type_hint: Option<Ty>,
    columns: &[String],
    placement: &str,
) -> anyhow::Result<Value> {
    let mut ctx = ExecutionContext::new();
    let mut options = CsvReadOptions::new();
    options.file_extension = std::path::Path::new(filename)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("");
    let df = ctx.read_csv(filename, options)?;
    let results = if !columns.is_empty() {
        let select_columns: Vec<&str> = columns.iter().map(String::as_str).collect();
        let df_filtered = df.select_columns(&select_columns).map_err(|_| {
            Error::Storage(format!(
                "these columns specified in the dataview are not in the dataset: {:?}",
                select_columns
            ))
        })?;
        df_filtered.collect().await?
    } else {
        df.collect().await?
    };
    let arr = datafusion_to_value(results, maybe_type_hint, placement)?;
    Ok(arr)
}

fn datafusion_to_value(
    results: Vec<RecordBatch>,
    maybe_type_hint: Option<Ty>,
    placement: &str,
) -> anyhow::Result<Value> {
    let column_types = &results
        .iter()
        .map(|x| {
            x.schema()
                .fields()
                .get(0)
                .map(|data| data.data_type().to_owned())
        })
        .collect::<Vec<_>>();

    let val_type = column_types
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("no columns found in datafusion dataframe"))?
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("could not infer type from CSV data"))?;

    let all_columns_same_type = column_types
        .iter()
        .all(|item| item.eq(&Some(val_type.clone())));

    if !all_columns_same_type {
        Err(anyhow::anyhow!(
            "loading CSV with mixed types (all types must be the same when loading into a tensor)"
        ))
    } else {
        let type_hint = maybe_type_hint.unwrap_or(Ty::HostFloat64Tensor);
        select_col_type(results, type_hint, placement).map_err(|e| {
            anyhow::anyhow!(
                "got unexpected type: expecting {} got {}: {}",
                type_hint,
                val_type,
                e
            )
        })
    }
}

fn select_col_type(
    results: Vec<RecordBatch>,
    type_hint: Ty,
    placement: &str,
) -> anyhow::Result<Value> {
    let plc = HostPlacement::from(placement);
    match type_hint {
        Ty::HostFloat64Tensor => {
            let ndarr = datafusion_to_ndarray::<arrow::datatypes::Float64Type>(results)?;
            let t: HostFloat64Tensor = plc.from_raw(ndarr);
            Ok(Value::from(t))
        }
        Ty::HostFloat32Tensor => {
            let ndarr = datafusion_to_ndarray::<arrow::datatypes::Float32Type>(results)?;
            let t: HostFloat32Tensor = plc.from_raw(ndarr);
            Ok(Value::from(t))
        }
        Ty::HostInt32Tensor => {
            let ndarr = datafusion_to_ndarray::<arrow::datatypes::Int32Type>(results)?;
            let t: HostInt32Tensor = plc.from_raw(ndarr);
            Ok(Value::from(t))
        }
        Ty::HostInt64Tensor => {
            let ndarr = datafusion_to_ndarray::<arrow::datatypes::Int64Type>(results)?;
            let t: HostInt64Tensor = plc.from_raw(ndarr);
            Ok(Value::from(t))
        }
        Ty::HostUint64Tensor => {
            let ndarr = datafusion_to_ndarray::<arrow::datatypes::UInt64Type>(results)?;
            let t: HostUint64Tensor = plc.from_raw(ndarr);
            Ok(Value::from(t))
        }
        Ty::HostUint32Tensor => {
            let ndarr = datafusion_to_ndarray::<arrow::datatypes::UInt32Type>(results)?;
            let t: HostUint32Tensor = plc.from_raw(ndarr);
            Ok(Value::from(t))
        }
        _ => Err(anyhow::anyhow!("variant not supported: {:?}", type_hint)),
    }
}

fn datafusion_to_ndarray<T>(results: Vec<RecordBatch>) -> anyhow::Result<Array2<T::Native>>
where
    T: arrow::datatypes::ArrowPrimitiveType,
{
    if let Some(first_result) = results.get(0) {
        let nrows = first_result.num_rows();
        let ncols = first_result.num_columns();

        // the output of datafusion::select_columns is column major (Fortran)
        // instead of row major (C, Rust, Python), so .f() from ArrayBuilder will construct
        // an ndarray from a column major vector
        let shape = (nrows, ncols).f();
        let mut as_vec: Vec<T::Native> = Vec::new();
        for col in first_result.columns().iter() {
            let array_downcast = col
                .as_any()
                .downcast_ref::<arrow::array::PrimitiveArray<T>>()
                .ok_or_else(|| anyhow::anyhow!("mismatched type"))?
                .values();
            as_vec.extend_from_slice(array_downcast);
        }
        let ndarr: Array2<T::Native> = Array2::from_shape_vec(shape, as_vec)?;
        Ok(ndarr)
    } else {
        Err(anyhow::anyhow!("no records found from query"))
    }
}
