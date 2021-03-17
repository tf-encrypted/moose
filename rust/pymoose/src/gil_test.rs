use pyo3::{prelude::*, types::{IntoPyDict, PyModule, PyBytes}};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;

#[test]
fn test_func() {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let comp_graph_py = PyModule::from_code(py, r#"
from moose import edsl
from moose.computation.utils import serialize_computation

alice = edsl.host_placement(name="alice")
bob = edsl.host_placement(name="bob")
carole = edsl.host_placement(name="carole")
dave = edsl.host_placement(name="dave")
rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

def f():
    @edsl.computation
    def my_comp():
        with alice:
            x = edsl.load("x", dtype=edsl.float64)
        with bob:
            y = edsl.load("y", dtype=edsl.float64)
        with rep:
            z1 = edsl.mul(x, y)

        with dave:
            res_dave = edsl.save("res", z1)

        return res_dave

    concrete_comp = edsl.trace(my_comp)
    with open("/tmp/computation.tmp", "wb") as f:
        f.write(serialize_computation(concrete_comp))

"#, "comp_graph.py", "comp_graph").unwrap();
    let _ = comp_graph_py.getattr("f").unwrap().call0().unwrap();
    let file = File::open("/tmp/computation.tmp").unwrap();

}