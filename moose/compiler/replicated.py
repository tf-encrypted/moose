# TODO(Morten) refactoring to not have this pass here
class ReplicatedPass:
    def run(self, computation, context):
        for op in computation.operations():
            print(op.placement_name)

        return computation
