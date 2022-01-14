use super::*;

impl SendOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _receiver: Role,
        _x: T,
    ) -> Result<Unit>
    where
        Value: From<T>,
    {
        // let x: Value = x.into();
        // sess.networking.send(&x, &receiver, &rendezvous_key)?;
        // Ok(Unit(plc.clone()))
        todo!()
    }
}

impl ReceiveOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _sender: Role,
    ) -> Result<T>
    where
        T: TryFrom<Value, Error = Error>,
        T: std::fmt::Debug,
        HostPlacement: PlacementPlace<S, T>,
    {
        // use std::convert::TryInto;
        // let value = sess.networking.receive(&sender, &rendezvous_key)?;
        // Ok(plc.place(sess, value.try_into()?))
        todo!()
    }

    pub(crate) fn missing_kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _sender: Role,
    ) -> Result<T>
    where
        T: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<T as KnownType<S>>::TY
        )))
    }
}

impl IdentityOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(sess: &S, plc: &HostPlacement, x: T) -> Result<T>
    where
        HostPlacement: PlacementPlace<S, T>,
    {
        let value = plc.place(sess, x);
        Ok(value)
    }

    pub(crate) fn missing_kernel<S: RuntimeSession, T>(_sess: &S, _plc: &HostPlacement, _x: T) -> Result<T>
    where
        T: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<T as KnownType<S>>::TY
        )))
    }
}

impl InputOp {
    pub(crate) fn kernel<S: RuntimeSession, O>(sess: &S, plc: &HostPlacement, arg_name: String) -> Result<O>
    where
        O: TryFrom<Value, Error = Error>,
        HostPlacement: PlacementPlace<S, O>,
    {
        use std::convert::TryInto;
        let value = sess
            .find_argument(&arg_name)
            .ok_or_else(|| Error::MissingArgument(arg_name.clone()))?;
        let value = plc.place(sess, value.try_into()?);
        Ok(value)
    }

    pub(crate) fn missing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _arg_name: String,
    ) -> Result<O>
    where
        O: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<O as KnownType<S>>::TY
        )))
    }
}

impl OutputOp {
    pub(crate) fn kernel<S: RuntimeSession, O>(sess: &S, plc: &HostPlacement, x: O) -> Result<O>
    where
        HostPlacement: PlacementPlace<S, O>,
    {
        // Output is not doing anything now, it is just a marker on the graph.
        // But it has to return a value because that's how we collect outputs in the old framework
        let x = plc.place(sess, x);
        Ok(x)
    }

    pub(crate) fn non_placing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        x: O,
    ) -> Result<O> {
        // Output is not doing anything now, it is just a marker on the graph.
        // But it has to return a value because that's how we collect outputs in the old framework
        Ok(x)
    }
}
