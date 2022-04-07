## About subcomputations

Moose runtime should be able to support launching subcomputations and waiting for their outputs in the main computation.

### Requirements and constraints

- Subcomputation is a DAG with multiple inputs and a *single (???)* output.
- Subcomputation has a name, which acts like a function name.
- Call to a subcomputation has three parts:
  - name
  - list of Constants to pass into a computation
  - list of input arguments to act as inputs
- Subcomputation is always executed on the same set of runners with the same role assignment as parent session.
- Recursion should be allowed. *(???)* Can we do that securely in the MPC setting?

### High-level design considerations.

- Each subcomputation gets a new Runtime Session. The new SessionID is generated using the original (top-level) session id appended with a random string and hashed.
- The top-level session id is passed in a separate field to every child session. This is both to generate new SessionID and for tracing.
- Launching subcomputation is a separate Operator (`Execute` ???), but it is not launched through the DispatchKernel. Instead it is handled similar
to the `Send/Receive` networking operations pair. Main reason for this is to be able to wait upon any number of inputs without implementing DispatchKernel of high arrity calls.

### Implementation details

First operations to benefit from Subcomputations should be operations iterating over the number of bits, such as `BitComposeOp`.
For example, `BitComposeOp` iterates over the bits `(0..RepRingT::BitLength::VALUE)` and for each bit index calls `IndexOp`, `RingInjectOp` and `AddOp` on a replicated placement.
The three operators will form a subcomputation.

Rust code for the kernel (draft):

```
impl BitComposeOp {
    pub(crate) fn rep_kernel(x)... {
        let sub_routine = named_subcomputation("single_bit_compose", |x, i| {
            let y = rep.index(sess, i, &x);
            rep.ring_inject(sess, i, y) // Result of the Subcomputation
        };
    
        let zeros = rep.fill(sess, 0u64.into(), &rep.shape(sess, &v[0]));
        let res = (0..RepRingT::BitLength::VALUE).fold(zeros, |x, i| {
            let y = sub_routine(sess, &x, Constant::try_from(i)?);
            rep.add(sess, &x, &y)
        };
        Ok(res)
    }
}
```

Textual format will be something like

```
// Defining a named subcomputation
SUB single_bit_compose {i: Ring128} (x: Tensor<Fixed128(24, 40)>)
    y = RepIndex{index = i}(x)
    ret = RepRingIject{index = i}(y)
RET ret

// Calling a named subcomputation
zeroes = RepFill{value = 0}(shape)

bit_0 = Execute {name = "single_bit_compose"} {i = 0} (x)
add_0 = RepAdd(zeroes, bit_0)

bit_1 = Execute {name = "single_bit_compose"} {i = 1} (x)
add_1 = RepAdd(add_0, bit_1)

bit_2 = Execute {name = "single_bit_compose"} {i = 2} (x)
add_2 = RepAdd(add_1, bit_2)

```

(In the example above we only save one operator per subcomputation call, which is not much and probably not worth it to launch a separate session,
but it is a small enough example to be great at illustrating the concept in the design doc).
