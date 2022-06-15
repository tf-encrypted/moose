Pymoose: Python bindings to the Elk compiler and Moose Runtime
===============
### Installation & Testing
```
pip install -r requirements-dev.txt
maturin develop
pytest -m "not slow" .
```

### Usage
More advanced examples can be found in the `examples/` directory.

```python
import numpy as np
import pymoose as pm

alice = pm.host_placement("alice")
bob = pm.host_placement("bob")
carole = pm.host_placement("carole")
encrypted = pm.replicated_placement("encrypted", [alice, bob, carole])

@pm.computation
def my_computation(
    x: pm.Argument(placement=alice, dtype=pm.float64),
    y: pm.Argument(placement=bob, dtype=pm.float64),
):

    with encrypted:
        w = pm.add(x, y)
        z = pm.mul(w, w)

    with carole:
        z = pm.output("z", z)

    return z

runtime = pm.GrpcMooseRuntime({
    alice: "localhost:50000",
    bob: "localhost:50001",
    carole: "localhost:50002",
})
runtime.set_default()

result, _ = my_computation(np.array(1.0), np.array(2.0))
result
# >>> {"z": array(9.0)}
result, _ = my_computation(np.array(2.0), np.array(3.0))
result
# >>> {"z", array(25.0)}
```
