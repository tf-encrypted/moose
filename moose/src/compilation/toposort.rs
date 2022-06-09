use crate::computation::Computation;
use crate::error::Error;

pub fn toposort(mut comp: Computation) -> anyhow::Result<Computation> {
    let graph = comp.as_graph();
    let toposort = petgraph::algo::toposort(&graph, None).map_err(|_| {
        Error::MalformedComputation("cycle detected in the computation graph".into())
    })?;

    // suppose we have [d, c, a, b] as our input computation
    // in toposort we will get [2, 3, 1, 0]
    let mut topo_map = vec![0; toposort.len()];
    for i in 0..topo_map.len() {
        topo_map[toposort[i].index()] = i
    }
    // topo_map will now have [3, 2, 0, 1] which corresponds to positions of [d, c, a, b] in the final toposorted comp
    // one can think of this vec as a map from an operation => corresponding position in the list of topological sorted ops

    // Next we are swapping values to their correct position until we hit the one meant for the current index
    // To understand, follow the example above:
    // for eg, with i = 0
    // [3, 2, 0, 1] => [1, 2, 0, 3] => [2, 1, 0, 3] => [0, 1, 2, 3]
    // [d, c, a, b] => [b, c, a, d] => [c, b, a, d] => [a, b, c, d]
    // when i = 1
    // [0, 1, 2, 3]
    // [a, b, c, d]
    // i = 2, 3; the ops stay the same as they are sorted
    // every element will be swapped at most once since at every point we are placing an element on its correct position

    for i in 0..comp.operations.len() {
        let mut next_pos = topo_map[i];
        while i != next_pos {
            comp.operations.swap(i, next_pos);
            topo_map.swap(i, next_pos);
            next_pos = topo_map[i];
        }
    }
    Ok(comp)
}
