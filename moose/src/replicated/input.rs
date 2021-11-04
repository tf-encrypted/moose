use crate::computation::{HostPlacement, InputOp, ReplicatedPlacement, Signature, Ty};
use crate::error::{Error, Result};
use crate::kernels::{PlacementInput, Session};
use crate::replicated::{AbstractReplicatedFixedTensor, AbstractReplicatedRingTensor};

impl InputOp {
    pub(crate) fn replicated_ring_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        arg_name: String,
    ) -> Result<AbstractReplicatedRingTensor<HostRingT>>
    where
        HostPlacement: PlacementInput<S, HostRingT>,
    {
        // TODO standardize this arg_name format for shares
        let lift_name =
            |player_ix, share_ix| format!("{0}/player{1}/share{2}", &arg_name, player_ix, share_ix);
        let (p0, p1, p2) = plc.host_placements();
        let in00 = p0.input(sess, lift_name(0, 0));
        let in10 = p0.input(sess, lift_name(0, 1));
        let in11 = p1.input(sess, lift_name(1, 1));
        let in21 = p1.input(sess, lift_name(1, 2));
        let in22 = p2.input(sess, lift_name(2, 2));
        let in02 = p2.input(sess, lift_name(2, 0));
        Ok(AbstractReplicatedRingTensor {
            shares: [[in00, in10], [in11, in21], [in22, in02]],
        })
    }

    pub(crate) fn replicated_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        sig: Signature,
        arg_name: String,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementInput<S, RepRingT>,
    {
        let ring_tensor = plc.input(sess, arg_name);

        let return_type = sig.ret();
        let ret_precision = match return_type {
            // TODO(jason,morten): figure out a good way to get this static type information
            //  from the Signature (improve Ty impl in values!)
            Ty::ReplicatedFixed64Tensor => Some((14, 23)),
            Ty::ReplicatedFixed128Tensor => Some((46, 40)),
            _ => None,
        };
        if ret_precision.is_none() {
            return Err(Error::TypeMismatch {
                expected: "ReplicatedFixedTensor".to_string(),
                found: return_type,
            });
        }
        let (integral_precision, fractional_precision) = ret_precision.unwrap();

        Ok(AbstractReplicatedFixedTensor {
            tensor: ring_tensor,
            integral_precision,
            fractional_precision,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computation::SessionId;
    use crate::host::{FromRawPlc, HostFloat32Tensor, HostRing64Tensor};
    use crate::replicated::{ReplicatedFixed64Tensor, ReplicatedRing64Tensor};
    use ndarray::{array, IxDyn};

    use crate::kernels::{
        PlacementFixedpointEncode, PlacementReveal, PlacementShareSetup, SyncSession,
    };

    #[test]
    fn test_rep_ring_input_op() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        // Create replicated input tensor in a previous session
        let sess0 = SyncSession::default();
        let setup = (*sess0.replicated_setup(&rep)).clone();
        let x = HostRing64Tensor::from_raw_plc(
            array![1u64, 2, 3].into_dimensionality::<IxDyn>().unwrap(),
            alice.clone(),
        );
        let x_shared = rep.share(&sess0, &setup, &x);

        // Populate test session args with shares of x
        let arg_name = "x".to_string();
        let lift_name = |name, player_ix, share_ix| {
            format!("{0}/player{1}/share{2}", name, player_ix, share_ix)
        };
        let mut new_args = std::collections::HashMap::new();
        for i in 0..3 {
            for j in 0..2 {
                new_args.insert(
                    lift_name(arg_name.clone(), i, (i + j) % 3),
                    x_shared.shares[i][j].clone().into(),
                );
            }
        }

        // Test input op
        let test_sess = SyncSession::new(SessionId::random(), new_args, Default::default());
        let y: ReplicatedRing64Tensor = rep.input(&test_sess, arg_name);
        let z = alice.reveal(&test_sess, &y);
        assert_eq!(x, z)
    }

    #[test]
    fn test_rep_fixed_input_op() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        // Create replicated input tensor in a previous session
        let sess0 = SyncSession::default();
        let setup = (*sess0.replicated_setup(&rep)).clone();
        let x = HostFloat32Tensor::from_raw_plc(
            array![1.0, 2.0, 3.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            alice.clone(),
        );
        // TODO change fixedpoint values when fixedpoint config is no longer hardcoded, see above TODO
        let x_encoded = alice.fixedpoint_encode(&sess0, 23, 14, &x);
        let x_shared = rep.share(&sess0, &setup, &x_encoded);

        // Populate test session args with shares of x
        let arg_name = "x".to_string();
        let lift_name = |name, player_ix, share_ix| {
            format!("{0}/player{1}/share{2}", name, player_ix, share_ix)
        };
        let mut new_args = std::collections::HashMap::new();
        for i in 0..3 {
            for j in 0..2 {
                new_args.insert(
                    lift_name(arg_name.clone(), i, (i + j) % 3),
                    x_shared.tensor.shares[i][j].clone().into(),
                );
            }
        }

        // Test input op
        let test_sess = SyncSession::new(SessionId::random(), new_args, Default::default());
        // TODO change fixedpoint values when fixedpoint config is no longer hardcoded, see above TODO
        let y: ReplicatedFixed64Tensor = rep.input(&test_sess, arg_name);
        let z = alice.reveal(&test_sess, &y);
        assert_eq!(x_encoded, z)
    }
}
