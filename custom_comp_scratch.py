## Do we want to have one class or two different classes depending if
## it's a local dry run or remote computation

mul = LocalMultiplicationTask(secret=False)
z = mul.run(x=np.array([1.0]), y=np.array([2.0]))
# When instantiating class, build local runtime
# Under run:
# - Seed x_onwer storage with x_data key and numpy value
# - Seed y_owner storage with y_data ky  and numpy value
# - Evaluate with arg {"x_uri": "x_data", "y_uri": "y_data", "x_query":None, "y_query":None, "out_uri": "out_data"}
# - get output from storage

# Remote production computation
job = RemoteMultiplicationTask(
        x_dataview=project.dataviews[0]["x"],
        y_dataview=project.dataviews[1]["y"],
        location_output=f"s3://{bucket}",
        output_owner=project.organizations[0].id
    )

## Is it possible to use the exact same computation for dry run and remote comp?
## the output of build_multiplcation should be traced/compile and store 
## in the coordinator for prod use case (Named computation approach)
def build_multiplication(local, x_placement_name, y_placement_name):
    alice = edsl.host_placement(name=x_placement_name)
    bob = edsl.host_placement(name=y_placement_name)
    cape = edsl.host_placement(name="cape-worker")
    repl = edsl.replicated_placement((alice, bob, cape))
    multiply_placement = repl

    @edsl.computation
    def other(
        x_uri: edsl.Argument(placement=alice, vtype=StringType()),
        x_query: edsl.Argument(placement=alice, vtype=StringType()),
        y_uri: edsl.Argument(placement=bob, vtype=StringType()),
        y_query: edsl.Argument(placement=bob, vtype=StringType()),
        out_uri: edsl.Argument(placement=alice, vtype=StringType())
    ):
        with alice:
            if local:
                x = load(x_uri)
            else:
                x = load(x_uri, x_query)

        with bob:
            if local:
                y = load(y_uri)
            else:
                y = load(y_uri, y_query)

        with multiply_placement:
            z = edsl.mul(x, y)

        with alice:
            out = edsl.save(out_uri, z)
        return 

    return other


# Remote custom computation
# job = MultiplicationTask(
#         x_dataview=project.dataviews[0]["x"],
#         y_dataview=project.dataviews[1]["y"],
#         location_output=f"s3://{bucket}",
#         output_owner=project.organizations[0].id
#     )

# class MultiplicationTask(MooseEdslMixin, GraphQLMixin):
#     def __init__(self, x_input, y_input, secret=False):
#         x_placement_name = super(GraphQLMixin).get_placement_name_from_organization(
#             x_input.owner
#         )
#         y_placement_name = super(GraphQLMixin).get_placement_name_from_organization(
#             y_input.owner
#         )
#         mult_edsl_func = build_multiplication(
#             secret=secret,
#             x_placement_name=x_placement_name,
#             y_placement_name=y_placement_name,
#         )
#         super().from_edsl_func(mult_edsl_func)


# def build_multiplication(secret, x_placement_name, y_placement_name):
#     alice = edsl.host_placement(name=x_placement_name)
#     bob = edsl.host_placement(name=y_placement_name)
#     if secret:
#         cape = edsl.host_placement(name="cape-worker")
#         repl = edsl.replicated_placement((alice, bob, cape))
#         multiply_placement = repl
#     else:
#         multiply_placement = alice
    
#     @edsl.computation
#     def other(
#         x_uri: edsl.Argument(placement=alice, vtype=StringType()),
#         y_uri: edsl.Argument(placement=bob, vtype=StringType()),
#     ) -> Tuple[
#         edsl.Result(bob),
#         edsl.Result(alice),
#         Tuple[edsl.Result(alice), edsl.Result(bob)],
#     ]:
#         with multiply_placement:
#             z = edsl.mul(x, y)
#         return x, y, z

#     return other


# Simple api to run local computation with Cape
alice = edsl.host_placement(name="alice")
bob = edsl.host_placement(name="bob")
cape = edsl.host_placement(name="cape-worker")
repl = edsl.replicated_placement((alice, bob, cape))
multiply_placement = repl

def my_computation(
    x_uri: edsl.Argument(placement=alice, vtype=StringType()),
    y_uri: edsl.Argument(placement=bob, vtype=StringType()),
    out_uri: edsl.Argument(placement=alice, vtype=StringType())
):

    with alice:
        x = edsl.load(x_uri)

    with bob:
        y = edsl.load(y_uri)
    
    with multiply_placement:
        z  = edsl.add(x, y)

    with  alice:
        out = edsl.save(out_uri, z)

    return out

cc = CustomComputation(
  computation=my_computation, # gets traced and compile under the hood
  placements={
    'alice': organizations[0],
    'bob': organizations[1]
  },
  arguments={
    'x_uri': 'x_path',
    'y_uri': 'y_path',
    'out_uri': 'out_path',
  }
)








