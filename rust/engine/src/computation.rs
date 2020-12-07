type Identifier = &'static str;

#[derive(Debug, Clone)]
pub enum Operator {
    Input(Identifier),
    Output(Identifier),
    Comm(CommOperator),
    Plain(Box<PlainOperator>), // TODO box?
    Ring(Box<RingOperator>),
    // Replicated(ReplicatedOperator),
}

#[derive(Debug, Clone)]
pub enum CommOperator {
    Send,
    Receive,
}

#[derive(Debug, Clone)]
pub enum PlainOperator {
    Constant(i32),
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
pub enum RingOperator {
    Add,
    Sub,
    Mul,
    Inv,
    SampleUniform,
}

impl Default for Operator {
    fn default() -> Self {
        Operator::Plain(Box::new(PlainOperator::Constant(0)))
    }
}

type Graph<N> = petgraph::Graph<N, ()>;
type Computation = Graph<Operator>;
type ValueMap = std::collections::HashMap<Identifier, i32>;

trait Executor {
    fn evaluate(&self, comp: &Computation, args: &ValueMap) -> ValueMap;
}

#[cfg(test)]
mod tests {
    use super::*;

    use my_executor::MyExecutor;

    use maplit::hashmap;

    fn create_example_computation() -> Computation {
        let mut graph = Computation::new();

        // add nodes
        let input_0 = graph.add_node(Operator::Input("input:0"));
        let input_1 = graph.add_node(Operator::Input("input:1"));
        let input_2 = graph.add_node(Operator::Input("input:2"));
        let input_3 = graph.add_node(Operator::Input("input:3"));
        let input_4 = graph.add_node(Operator::Input("input:4"));
        let input_5 = graph.add_node(Operator::Input("input:5"));
        let add_0 = graph.add_node(Operator::Add);
        let add_1 = graph.add_node(Operator::Add);
        let add_2 = graph.add_node(Operator::Add);
        let add_3 = graph.add_node(Operator::Add);
        let add_4 = graph.add_node(Operator::Add);
        let mul_0 = graph.add_node(Operator::Mul);
        let output = graph.add_node(Operator::Output("output"));

        // add edges
        graph.extend_with_edges(&[
            (input_0, add_0),
            (input_1, add_0),
            (input_2, add_1),
            (input_3, add_1),
            (input_4, add_2),
            (input_5, add_3),
            (add_0, mul_0),
            (add_1, mul_0),
            (mul_0, add_2),
            (mul_0, add_3),
            (add_2, add_4),
            (add_3, add_4),
            (add_4, output),
        ]);

        graph
    }

    #[test]
    fn it_works() {
        let executor = MyExecutor::new();

        let comp = create_example_computation();

        let args = hashmap! {
            "input:0" => 1,
            "input:1" => 2,
            "input:2" => 3,
            "input:3" => 4,
            "input:4" => 5,
            "input:5" => 6,
        };
        let res = executor.evaluate(&comp, &args);
        assert_eq!(
            res,
            hashmap! {
                "output" => 53,
            }
        );
    }
}
