use super::*;
use crate::computation::{InputOp, Signature, Ty};
use crate::error::{Error, Result};
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::kernels::PlacementInput;
use crate::replicated::aes::RepAesKey;
use crate::{N128, N224, N64};
use std::marker::PhantomData;

impl InputOp {
    pub(crate) fn replicated_ring_kernel<S: Session, HostTensorT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        arg_name: String,
    ) -> Result<RepTensor<HostTensorT>>
    where
        HostPlacement: PlacementInput<S, HostTensorT>,
    {
        // TODO standardize this arg_name format for shares
        let repl_roles = &plc.owners;
        let lift_name = |player_ix, share_ix| {
            let repl_role: &Role = &repl_roles[player_ix];
            format!("{0}/{1}/share{2}", &arg_name, repl_role.0, share_ix)
        };
        let (p0, p1, p2) = plc.host_placements();
        let in00 = p0.input(sess, lift_name(0, 0));
        let in10 = p0.input(sess, lift_name(0, 1));
        let in11 = p1.input(sess, lift_name(1, 1));
        let in21 = p1.input(sess, lift_name(1, 2));
        let in22 = p2.input(sess, lift_name(2, 2));
        let in02 = p2.input(sess, lift_name(2, 0));
        Ok(RepTensor {
            shares: [[in00, in10], [in11, in21], [in22, in02]],
        })
    }

    pub(crate) fn replicated_bitarray64<S: Session, RepBitTensorT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        arg_name: String,
    ) -> Result<RepBitArray<RepBitTensorT, N64>>
    where
        ReplicatedPlacement: PlacementInput<S, RepBitTensorT>,
    {
        // TODO(Morten) ideally we should verify that shape of bit tensor
        let bit_tensor = plc.input(sess, arg_name);
        Ok(RepBitArray(bit_tensor, PhantomData))
    }

    pub(crate) fn replicated_bitarray128<S: Session, RepBitTensorT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        arg_name: String,
    ) -> Result<RepBitArray<RepBitTensorT, N128>>
    where
        ReplicatedPlacement: PlacementInput<S, RepBitTensorT>,
    {
        // TODO(Morten) ideally we should verify that shape of bit tensor
        let bit_tensor = plc.input(sess, arg_name);
        Ok(RepBitArray(bit_tensor, PhantomData))
    }

    pub(crate) fn replicated_bitarray224<S: Session, RepBitTensorT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        arg_name: String,
    ) -> Result<RepBitArray<RepBitTensorT, N224>>
    where
        ReplicatedPlacement: PlacementInput<S, RepBitTensorT>,
    {
        // TODO(Morten) ideally we should verify that shape of bit tensor
        let bit_tensor = plc.input(sess, arg_name);
        Ok(RepBitArray(bit_tensor, PhantomData))
    }

    pub(crate) fn replicated_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        sig: Signature,
        arg_name: String,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementInput<S, RepRingT>,
    {
        let ring_tensor = plc.input(sess, arg_name);

        let (integral_precision, fractional_precision) = match sig.ret() {
            // TODO(jason,morten): figure out a good way to get this static type information
            //  from the Signature (improve Ty impl in values!)
            Ty::ReplicatedFixed64Tensor => Ok((14, 23)),
            Ty::ReplicatedFixed128Tensor => Ok((24, 40)),
            _ => Err(Error::TypeMismatch {
                expected: "ReplicatedFixedTensor".to_string(),
                found: sig.ret(),
            }),
        }?;

        Ok(RepFixedTensor {
            tensor: ring_tensor,
            integral_precision,
            fractional_precision,
        })
    }

    pub(crate) fn replicated_aes_kernel<S: Session, RepBitArrayT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        arg_name: String,
    ) -> Result<RepAesKey<RepBitArrayT>>
    where
        ReplicatedPlacement: PlacementInput<S, RepBitArrayT>,
    {
        let rep_bit_array = plc.input(sess, arg_name);
        Ok(RepAesKey(rep_bit_array))
    }
}

#[cfg(feature = "sync_execute")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::computation::SessionId;
    use crate::kernels::{PlacementFixedpointEncode, PlacementReveal, PlacementShare};
    use crate::prelude::*;
    use crate::storage::local::LocalSyncStorage;
    use ndarray::prelude::*;
    use std::rc::Rc;

    #[test]
    fn test_input_rep_ring() {
        let alice = HostPlacement::from("alice");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        // Create replicated input tensor in a previous session
        let sess0 = SyncSession::default();
        let x: HostRing64Tensor = alice.from_raw(array![1u64, 2, 3]);
        let x_shared = rep.share(&sess0, &x);

        // Populate test session args with shares of x
        let arg_name = "x".to_string();
        let repl_roles = &rep.owners;
        let lift_name = |player_ix, share_ix| {
            let repl_role: &Role = &repl_roles[player_ix];
            format!("{0}/{1}/share{2}", &arg_name, repl_role.0, share_ix)
        };
        let mut new_args = std::collections::HashMap::new();
        for i in 0..3 {
            for j in 0..2 {
                new_args.insert(
                    lift_name(i, (i + j) % 3),
                    x_shared.shares[i][j].clone().into(),
                );
            }
        }

        // Test input op
        let test_sess = SyncSession::from_storage(
            SessionId::random(),
            new_args,
            Default::default(),
            Rc::new(LocalSyncStorage::default()),
        );
        let y: ReplicatedRing64Tensor = rep.input(&test_sess, arg_name);
        let z = alice.reveal(&test_sess, &y);
        assert_eq!(x, z)
    }

    #[test]
    fn test_input_rep_fixed() {
        let alice = HostPlacement::from("alice");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        // Create replicated input tensor in a previous session
        let sess0 = SyncSession::default();
        let x: HostFloat32Tensor = alice.from_raw(array![1.0, 2.0, 3.0]);
        // TODO change fixedpoint values when fixedpoint config is no longer hardcoded, see above TODO
        let x_encoded = alice.fixedpoint_encode(&sess0, 23, 14, &x);
        let x_shared = rep.share(&sess0, &x_encoded);

        // Populate test session args with shares of x
        let arg_name = "x".to_string();
        let repl_roles = &rep.owners;
        let lift_name = |player_ix, share_ix| {
            let repl_role: &Role = &repl_roles[player_ix];
            format!("{0}/{1}/share{2}", &arg_name, repl_role.0, share_ix)
        };
        let mut new_args = std::collections::HashMap::new();
        for i in 0..3 {
            for j in 0..2 {
                new_args.insert(
                    lift_name(i, (i + j) % 3),
                    x_shared.tensor.shares[i][j].clone().into(),
                );
            }
        }

        // Test input op
        let test_sess = SyncSession::from_storage(
            SessionId::random(),
            new_args,
            Default::default(),
            Rc::new(LocalSyncStorage::default()),
        );
        // TODO change fixedpoint values when fixedpoint config is no longer hardcoded, see above TODO
        let y: ReplicatedFixed64Tensor = rep.input(&test_sess, arg_name);
        let z = alice.reveal(&test_sess, &y);
        assert_eq!(x_encoded, z)
    }
}
