import json

from pymoose import edsl

from . import model
from . import model_utils as utils


class DecisionTreeRegressor(model.AesPredictor):
    def __init__(self, weights, children, split_conditions, split_indices):
        super().__init__()
        self.weights = weights
        self.left, self.right = children
        self.split_conditions = split_conditions
        self.split_indices = split_indices

    @classmethod
    def from_json(cls, tree_json):
        weights = dict(enumerate(tree_json["base_weights"]))
        left = _map_json_to_onnx_leaves(tree_json["left_children"])
        right = _map_json_to_onnx_leaves(tree_json["right_children"])
        split_conditions = tree_json["split_conditions"]
        split_indices = tree_json["split_indices"]
        return cls(weights, (left, right), split_conditions, split_indices)

    def predictor_factory(
        self,
        nb_features,
        rescale_factor=1.0,
        fixedpoint_dtype=utils.DEFAULT_FIXED_DTYPE,
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
        leaf_weights = {ix: rescale_factor * w for ix, w in self.weights.items()}
        features_vec = [edsl.index_axis(x, axis=1, index=i) for i in range(nb_features)]
        return self._traverse_tree(0, leaf_weights, features_vec, fixedpoint_dtype)

    def _traverse_tree(self, node, leaf_weights, x_features, fixedpoint_dtype):
        left_child = self.left[node]
        right_child = self.right[node]
        if left_child != 0 and right_child != 0:
            # we're at an inner node; this is the recursive case
            selector = edsl.less(
                x_features[self.split_indices[node]],
                self.fixedpoint_constant(
                    self.split_conditions[node], self.mirrored, dtype=fixedpoint_dtype
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
            assert left_child == 0
            assert right_child == 0
            return self.fixedpoint_constant(leaf_weights[node], self.mirrored)


class TreeEnsembleRegressor(model.AesPredictor):
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

    @classmethod
    def from_onnx(cls, model_proto):
        forest_node = utils.find_node_in_model_proto(
            model_proto, "TreeEnsembleRegressor", enforce=False
        )
        if forest_node is None:
            raise ValueError(
                "Incompatible ONNX graph provided: graph must contain a "
                "TreeEnsembleRegressor operator."
            )

        # construct `trees` argument
        nodes_treeids_attr = utils.find_attribute_in_node(forest_node, "nodes_treeids")
        assert nodes_treeids_attr.type == 7  # INTS
        nodes_treeids = nodes_treeids_attr.ints

        left_attr = utils.find_attribute_in_node(forest_node, "nodes_truenodeids")
        assert left_attr.type == 7
        left = left_attr.ints

        right_attr = utils.find_attribute_in_node(forest_node, "nodes_falsenodeids")
        assert right_attr.type == 7
        right = right_attr.ints

        split_conditions_attr = utils.find_attribute_in_node(
            forest_node, "nodes_values"
        )
        assert split_conditions_attr.type == 6
        split_conditions = split_conditions_attr.floats

        split_indices_attr = utils.find_attribute_in_node(
            forest_node, "nodes_featureids"
        )
        assert split_indices_attr.type == 7
        split_indices = split_indices_attr.ints

        target_nodeids_attr = utils.find_attribute_in_node(
            forest_node, "target_nodeids"
        )
        assert target_nodeids_attr.type == 7
        target_nodeids = target_nodeids_attr.ints

        target_treeids_attr = utils.find_attribute_in_node(
            forest_node, "target_treeids"
        )
        assert target_treeids_attr.type == 7
        target_treeids = target_treeids_attr.ints

        target_weights_attr = utils.find_attribute_in_node(
            forest_node, "target_weights"
        )
        assert target_weights_attr.type == 6  # FLOATS
        target_weights = target_weights_attr.floats

        nb_trees = len(set(nodes_treeids))
        tree_args = [
            {
                "weights": {},
                "children": [[], []],
                "split_indices": [],
                "split_conditions": [],
            }
            for _ in range(nb_trees)
        ]

        for i, tree_id in enumerate(nodes_treeids):
            tree_args[tree_id]["children"][0].append(left[i])
            tree_args[tree_id]["children"][1].append(right[i])
            tree_args[tree_id]["split_indices"].append(split_indices[i])
            tree_args[tree_id]["split_conditions"].append(split_conditions[i])

        for i, tree_id in enumerate(target_treeids):
            tree_args[tree_id]["weights"][target_nodeids[i]] = target_weights[i]

        trees = [DecisionTreeRegressor(**kwargs) for kwargs in tree_args]

        # `nb_features` arg
        model_input = model_proto.graph.input[0]
        input_shape = utils.find_input_shape(model_input)
        assert len(input_shape) == 2
        nb_features = input_shape[1].dim_value

        # `base_score` arg
        base_score_attr = utils.find_attribute_in_node(forest_node, "base_values")
        assert base_score_attr.type == 6  # FLOATS
        base_score = base_score_attr.floats[0]

        # `learning_rate` arg
        # NOTE: ONNX assumes the leaf weights have already been scaled by the
        # learning rate, so we keep our forest's learning_rate scaled fixed at 1.0
        learning_rate = 1.0

        return cls(trees, nb_features, base_score, learning_rate)

    def predictor_factory(self, fixedpoint_dtype=utils.DEFAULT_FIXED_DTYPE):
        # TODO[jason] make it more ergonomic for edsl.computation to bind args during
        #   tracing w/ edsl.trace
        @edsl.computation
        def predictor(
            aes_data: edsl.Argument(
                self.alice, vtype=edsl.AesTensorType(dtype=fixedpoint_dtype)
            ),
            aes_key: edsl.Argument(self.replicated, vtype=edsl.AesKeyType()),
        ):
            x = self.handle_aes_input(aes_key, aes_data, decryptor=self.replicated)
            with self.replicated:
                y = self._forest_fn(x, fixedpoint_dtype=fixedpoint_dtype)
            return self.handle_output(y, prediction_handler=self.bob)

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
            self.base_score, self.mirrored, dtype=fixedpoint_dtype
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
            DecisionTreeRegressor.from_json(tree)
            for tree in model_json["learner"]["gradient_booster"]["model"]["trees"]
        ]
        return trees, nb_features, base_score, learning_rate


def _map_json_to_onnx_leaves(json_leaves):
    return [0 if child == -1 else child for child in json_leaves]
