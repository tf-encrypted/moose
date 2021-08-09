use crate::computation::*;

/// Lower fixedpoint ops into ring ops on HostPlacement.
pub fn host_ring_lowering(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    let ops: anyhow::Result<Vec<Operation>> = comp.operations.iter().map(lower).collect();
    Ok(Some(Computation { operations: ops? }))
}

fn lower(op: &Operation) -> anyhow::Result<Operation> {
    match op {
        Operation {
            placement: Placement::Host(host),
            kind: Operator::FixedpointDecode(FixedpointDecodeOp { sig, precision }),
            name,
            inputs,
        } => Ok(Operation {
            name: name.clone(),
            kind: Operator::FixedpointRingDecode(FixedpointRingDecodeOp {
                sig: Signature::unary(mutate_ty(sig.arg(0)?), mutate_ty(sig.ret())),
                scaling_base: 2,
                scaling_exp: *precision,
            }),
            inputs: inputs.clone(),
            placement: host.into(),
        }),
        Operation {
            placement: Placement::Host(host),
            kind: Operator::FixedpointEncode(FixedpointEncodeOp { sig, precision }),
            name,
            inputs,
        } => Ok(Operation {
            name: name.clone(),
            kind: Operator::FixedpointRingEncode(FixedpointRingEncodeOp {
                sig: Signature::unary(mutate_ty(sig.arg(0)?), mutate_ty(sig.ret())),
                scaling_base: 2,
                scaling_exp: *precision,
            }),
            inputs: inputs.clone(),
            placement: host.into(),
        }),
        // TODO: Which other Ops we need to support lowering here?
        // On the Python side we have: Add, Mul, Sub, Sum, Trunc. The later implemented as RingShr (!).
        _ => Ok(op.clone()),
    }
}

fn mutate_ty(ty: Ty) -> Ty {
    match ty {
        Ty::Fixed128Tensor => Ty::Ring128Tensor,
        _ => ty,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_computation::ToTextual;
    use std::convert::TryInto;

    #[test]
    fn test_no_changes() -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])} @Host(alice)
        y = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])} @Host(alice)
        mul = StdMul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)
        dot = StdDot: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)
        mean = StdMean: (Float32Tensor) -> Float32Tensor (dot) @Host(alice)"#;

        let comp = host_ring_lowering(&source.try_into()?)?
            .unwrap()
            .to_textual();
        // Lowering should not introduce any changes to such a computation
        assert!(comp.contains(
            "mul = StdMul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)"
        ));
        assert!(comp.contains(
            "dot = StdDot: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)"
        ));
        assert!(
            comp.contains("mean = StdMean: (Float32Tensor) -> Float32Tensor (dot) @Host(alice)")
        );
        Ok(())
    }

    #[test]
    fn test_encode_decode() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x = FixedpointEncode{precision=27}: (Float64Tensor) -> Fixed128Tensor (a) @Host(alice)
        y = FixedpointDecode{precision=27}: (Fixed128Tensor) -> Float64Tensor (b) @Host(alice)
        "#;

        let comp = host_ring_lowering(&source.try_into()?)?
            .unwrap()
            .to_textual();
        // Lowering should change encode and decode
        assert!(comp.contains(
            "x = FixedpointRingEncode{scaling_base=2, scaling_exp=27}: (Float64Tensor) -> Ring128Tensor (a) @Host(alice)"
        ));
        assert!(comp.contains(
            "y = FixedpointRingDecode{scaling_base=2, scaling_exp=27}: (Ring128Tensor) -> Float64Tensor (b) @Host(alice)"
        ));
        Ok(())
    }
}
