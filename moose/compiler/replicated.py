# TODO(Morten) refactoring to not have this pass here
class ReplicatedPass:
    def run(self, computation, context):
        for op in computation.operations.values():
            print(op.placement_name)

        performed_changes = False
        return computation, performed_changes
