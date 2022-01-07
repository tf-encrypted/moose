import abc
import json

from pymoose import edsl


class StandardModel(metaclass=abc.ABCMeta):
    def __init__(self):
        (
            (self.alice, self.bob, self.carole),
            self.replicated,
        ) = self._standard_replicated_placements()

    def _standard_replicated_placements(self):
        alice = edsl.host_placement("alice")
        bob = edsl.host_placement("bob")
        carole = edsl.host_placement("carole")
        replicated = edsl.replicated_placement(
            name="replicated", players=[alice, bob, carole]
        )
        return (alice, bob, carole), replicated

    @property
    def host_placements(self):
        return self.alice, self.bob, self.carole

    @classmethod
    def fixedpoint_constant(cls, x, plc, dtype=edsl.fixed(8, 27)):
        x = edsl.constant(x, dtype=edsl.float64, placement=plc)
        return edsl.cast(x, dtype=dtype, placement=plc)

    @abc.abstractmethod
    def predictor_factory(self, *args, **kwargs):
        pass


class XGBoostTreeRegressor(StandardModel):
    def __init__(self, weights, children, split_conditions, split_indices):
        super().__init__()
        self.weights = weights
        self.left, self.right = children
        self.split_conditions = split_conditions
        self.split_indices = split_indices

    @classmethod
    def from_json(cls, tree_json):
        weights = tree_json["base_weights"]
        left = tree_json["left_children"]
        right = tree_json["right_children"]
        split_conditions = tree_json["split_conditions"]
        split_indices = tree_json["split_indices"]
        return cls(weights, (left, right), split_conditions, split_indices)

    def predictor_factory(
        self, nb_features, rescale_factor=1.0, fixedpoint_dtype=edsl.fixed(8, 27),
    ):
        # TODO[jason] make it more ergonomic for edsl.computation to bind args during
        #   tracing w/ edsl.trace
        @edsl.computation
        def predictor(x: edsl.Argument(self.alice, dtype=edsl.float64)):
            with self.alice:
                x = edsl.cast(x, dtype=fixedpoint_dtype)

            with self.replicated:
                y = self._tree_fn(
                    x,
                    nb_features,
                    rescale_factor=rescale_factor,
                    fixedpoint_dtype=fixedpoint_dtype,
                )

            with self.bob:
                y = edsl.cast(y, dtype=edsl.float64)

            return y

        return predictor

    def _tree_fn(self, x, nb_features, rescale_factor, fixedpoint_dtype):
        leaf_weights = [rescale_factor * w for w in self.weights]
        features_vec = [edsl.index_axis(x, axis=1, index=i) for i in range(nb_features)]
        return self._traverse_tree(0, leaf_weights, features_vec, fixedpoint_dtype)

    def _traverse_tree(self, node, leaf_weights, x_features, fixedpoint_dtype):
        left_child = self.left[node]
        right_child = self.right[node]
        if left_child != -1 and right_child != -1:
            # we're at an inner node; this is the recursive case
            selector = edsl.less(
                x_features[self.split_indices[node]],
                self.fixedpoint_constant(
                    self.split_conditions[node], self.alice, dtype=fixedpoint_dtype
                ),
            )
            return edsl.mux(
                selector,
                self._traverse_tree(
                    left_child, leaf_weights, x_features, fixedpoint_dtype
                ),
                self._traverse_tree(
                    right_child, leaf_weights, x_features, fixedpoint_dtype
                ),
            )
        else:
            assert left_child == -1
            assert right_child == -1
            return self.fixedpoint_constant(leaf_weights[node], self.alice)


class XGBoostForestRegressor(StandardModel):
    def __init__(self, trees, nb_features, base_score, learning_rate):
        super().__init__()
        self.nb_features = nb_features
        self.trees = trees
        self.base_score = base_score
        self.learning_rate = learning_rate

    @classmethod
    def from_json(cls, model_json):
        forest_args = cls._unbundle_forest(model_json)
        return cls(*forest_args)

    def predictor_factory(self, fixedpoint_dtype=edsl.fixed(8, 27)):
        # TODO[jason] make it more ergonomic for edsl.computation to bind args during
        #   tracing w/ edsl.trace
        @edsl.computation
        def predictor(x: edsl.Argument(self.alice, dtype=edsl.float64)):
            with self.alice:
                x = edsl.cast(x, dtype=fixedpoint_dtype)

            with self.replicated:
                y = self._forest_fn(x, fixedpoint_dtype=fixedpoint_dtype)

            with self.bob:
                y = edsl.cast(y, dtype=edsl.float64)

            return y

        return predictor

    def _forest_fn(self, x, fixedpoint_dtype):
        tree_scores = [
            tree._tree_fn(
                x,
                self.nb_features,
                rescale_factor=self.learning_rate,
                fixedpoint_dtype=fixedpoint_dtype,
            )
            for tree in self.trees
        ]
        final_score = self.fixedpoint_constant(
            self.base_score, self.alice, dtype=fixedpoint_dtype
        )
        for tree_score in tree_scores:
            final_score = edsl.add(tree_score, final_score)
        return final_score

    @classmethod
    def _unbundle_forest_params(cls, model_json):
        nb_features = int(model_json["learner"]["learner_model_param"]["num_feature"])
        base_score = float(model_json["learner"]["learner_model_param"]["base_score"])
        learning_rate = json.loads(model_json["learner"]["attributes"]["scikit_learn"])[
            "learning_rate"
        ]
        return nb_features, base_score, learning_rate

    @classmethod
    def _unbundle_forest(cls, model_json):
        nb_features, base_score, learning_rate = cls._unbundle_forest_params(model_json)
        trees = [
            XGBoostTreeRegressor.from_json(tree)
            for tree in model_json["learner"]["gradient_booster"]["model"]["trees"]
        ]
        return trees, nb_features, base_score, learning_rate
