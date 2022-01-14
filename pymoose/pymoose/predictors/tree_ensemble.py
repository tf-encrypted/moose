import abc
import json

from pymoose import edsl
from pymoose.predictors import aes_predictor
from pymoose.predictors import predictor_utils as utils


class DecisionTreeRegressor(aes_predictor.AesPredictor):
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

    def predictor_factory(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} is not meant to be used directly as an "
            "AesPredictor model. Consider expressing your decision tree as a tree "
            "ensemble with another AesPredictor implementation."
        )

    def __call__(self, x, n_features, rescale_factor, fixedpoint_dtype):
        leaf_weights = {ix: rescale_factor * w for ix, w in self.weights.items()}
        features_vec = [edsl.index_axis(x, axis=1, index=i) for i in range(n_features)]
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
            return self.fixedpoint_constant(leaf_weights[node], self.carole)


class TreeEnsemble(aes_predictor.AesPredictor, metaclass=abc.ABCMeta):
    def __init__(self, trees, n_features, base_score, learning_rate):
        super().__init__()
        self.n_features = n_features
        self.trees = trees
        self.base_score = base_score
        self.learning_rate = learning_rate

    @classmethod
    @abc.abstractmethod
    def from_onnx(cls, model_proto):
        pass

    @abc.abstractmethod
    def post_transform(self, tree_scores, fixedpoint_dtype):
        pass

    def forest_fn(self, x, fixedpoint_dtype):
        return [
            tree(
                x,
                self.n_features,
                rescale_factor=self.learning_rate,
                fixedpoint_dtype=fixedpoint_dtype,
            )
            for tree in self.trees
        ]

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
                tree_scores = self.forest_fn(x, fixedpoint_dtype=fixedpoint_dtype)
                y = self.post_transform(tree_scores, fixedpoint_dtype=fixedpoint_dtype)
            return self.handle_output(y, prediction_handler=self.bob)

        return predictor

    @classmethod
    def inner_onnx(cls, model_proto, forest_node_name):
        forest_node = utils.find_node_in_model_proto(
            model_proto, forest_node_name, enforce=False
        )
        if forest_node is None:
            raise ValueError(
                "Incompatible ONNX graph provided: graph must contain a "
                f"{forest_node_name} operator."
            )

        # construct `tree_args` for `trees` argument
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

        tree_args = (nodes_treeids, left, right, split_conditions, split_indices)

        n_trees = len(set(nodes_treeids))

        # `n_features` arg
        model_input = model_proto.graph.input[0]
        input_shape = utils.find_input_shape(model_input)
        assert len(input_shape) == 2
        n_features = input_shape[1].dim_value

        # `base_score` arg
        base_score_attr = utils.find_attribute_in_node(
            forest_node, "base_values", enforce=False
        )
        if base_score_attr is None:
            base_score = 0.0
        else:
            assert base_score_attr.type == 6  # FLOATS
            base_score = base_score_attr.floats[0]

        # `learning_rate` arg
        # NOTE: ONNX assumes the leaf weights have already been scaled by the
        # learning rate, so we keep our forest's learning_rate scaled fixed at 1.0
        learning_rate = 1.0

        return forest_node, tree_args, n_trees, n_features, base_score, learning_rate


class TreeEnsembleClassifier(TreeEnsemble):
    def __init__(
        self, trees, n_features, base_score, learning_rate, n_classes, tree_classes
    ):
        super().__init__(trees, n_features, base_score, learning_rate)
        self.n_classes = n_classes
        self.tree_classes = tree_classes

    @classmethod
    def from_onnx(cls, model_proto):
        (
            forest_node,
            (nodes_treeids, left, right, split_conditions, split_indices),
            n_trees,
            n_features,
            base_score,
            learning_rate,
        ) = cls.inner_onnx(model_proto, "TreeEnsembleClassifier")

        class_ids_attr = utils.find_attribute_in_node(forest_node, "class_ids")
        assert class_ids_attr.type == 7
        class_ids = class_ids_attr.ints

        class_nodeids_attr = utils.find_attribute_in_node(forest_node, "class_nodeids")
        assert class_nodeids_attr.type == 7
        class_nodeids = class_nodeids_attr.ints

        class_treeids_attr = utils.find_attribute_in_node(forest_node, "class_treeids")
        assert class_treeids_attr.type == 7
        class_treeids = class_treeids_attr.ints

        class_weights_attr = utils.find_attribute_in_node(forest_node, "class_weights")
        assert class_weights_attr.type == 6
        class_weights = class_weights_attr.floats

        classlabels_ints = utils.find_attribute_in_node(
            forest_node, "classlabels_int64s", enforce=False
        )
        classlabels_strings = utils.find_attribute_in_node(
            forest_node, "classlabels_strings", enforce=False
        )
        assert classlabels_ints is not None or classlabels_strings is not None
        if classlabels_ints is not None:
            classlabels = classlabels_ints.ints
        elif classlabels_strings is not None:
            classlabels = classlabels_strings.strings
        n_classes = len(classlabels)

        tree_args = [
            {
                "weights": {},
                "children": [[], []],
                "split_indices": [],
                "split_conditions": [],
            }
            for _ in range(n_trees)
        ]

        for i, tree_id in enumerate(nodes_treeids):
            tree_args[tree_id]["children"][0].append(left[i])
            tree_args[tree_id]["children"][1].append(right[i])
            tree_args[tree_id]["split_indices"].append(split_indices[i])
            tree_args[tree_id]["split_conditions"].append(split_conditions[i])

        for i, tree_id in enumerate(class_treeids):
            tree_args[tree_id]["weights"][class_nodeids[i]] = class_weights[i]

        trees = [DecisionTreeRegressor(**kwargs) for kwargs in tree_args]
        tree_class_map = {
            tree_id: class_id for tree_id, class_id in zip(class_treeids, class_ids)
        }

        return cls(
            trees, n_features, base_score, learning_rate, n_classes, tree_class_map
        )

    def post_transform(self, tree_scores, fixedpoint_dtype):
        if self.n_classes == 2:
            return self._double_sigmoid(tree_scores, fixedpoint_dtype)
        else:
            return self._ovr_softmax(tree_scores, axis=1)

    def _double_sigmoid(self, tree_scores, fixedpoint_dtype):
        logit = edsl.add_n(tree_scores)
        pos_prob = edsl.sigmoid(logit)
        # TODO match binary classification format from sklearn, etc.
        # one = self.fixedpoint_constant(1, plc=self.mirrored, dtype=fixedpoint_dtype)
        # neg_prob = edsl.sub(one, pos_prob)
        # return edsl.concatenate([neg_prob, pos_prob], axis=1)
        return pos_prob

    def _ovr_softmax(self, tree_scores, axis):
        ovr_results = [[] for _ in range(self.n_classes)]
        for tree_ix, model_ix in self.tree_classes.items():
            ovr_results[model_ix].append(tree_scores[tree_ix])
        ovr_logits = [edsl.expand_dims(edsl.add_n(ovr), axis=1) for ovr in ovr_results]
        logit = edsl.concatenate(ovr_logits, axis=1)
        return self._temp_softmax(logit, axis=1)

    def _temp_softmax(self, x, axis):
        # TODO replace with edsl.max(x, axis)
        x_bound = edsl.sub(x, edsl.sum(x, axis))
        x_exp = edsl.exp(x_bound)
        return edsl.div(x_exp, edsl.sum(x_exp, axis))


class TreeEnsembleRegressor(TreeEnsemble):
    @classmethod
    def from_json(cls, model_json):
        forest_args = cls._unbundle_forest(model_json)
        return cls(*forest_args)

    @classmethod
    def from_onnx(cls, model_proto):
        (
            forest_node,
            (nodes_treeids, left, right, split_conditions, split_indices),
            n_trees,
            n_features,
            base_score,
            learning_rate,
        ) = cls.inner_onnx(model_proto, "TreeEnsembleRegressor")

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

        tree_args = [
            {
                "weights": {},
                "children": [[], []],
                "split_indices": [],
                "split_conditions": [],
            }
            for _ in range(n_trees)
        ]

        for i, tree_id in enumerate(nodes_treeids):
            tree_args[tree_id]["children"][0].append(left[i])
            tree_args[tree_id]["children"][1].append(right[i])
            tree_args[tree_id]["split_indices"].append(split_indices[i])
            tree_args[tree_id]["split_conditions"].append(split_conditions[i])

        for i, tree_id in enumerate(target_treeids):
            tree_args[tree_id]["weights"][target_nodeids[i]] = target_weights[i]

        trees = [DecisionTreeRegressor(**kwargs) for kwargs in tree_args]

        return cls(trees, n_features, base_score, learning_rate)

    def post_transform(self, tree_scores, fixedpoint_dtype):
        base_score = self.fixedpoint_constant(
            self.base_score, self.carole, dtype=fixedpoint_dtype
        )
        # ugly way of ensuring it's replicated;
        # normally it would be replicated just by using it in add_n @ replicated,
        # but here it's input to an op w/ variadic signature, which does not do
        # the work of converting from host to replicated for all of its inputs
        base_score = edsl.identity(base_score)
        return edsl.add_n([base_score] + tree_scores)

    @classmethod
    def _unbundle_forest(cls, model_json):
        n_features, base_score, learning_rate = cls._unbundle_forest_params(model_json)
        trees = [
            DecisionTreeRegressor.from_json(tree)
            for tree in model_json["learner"]["gradient_booster"]["model"]["trees"]
        ]
        return trees, n_features, base_score, learning_rate

    @classmethod
    def _unbundle_forest_params(cls, model_json):
        n_features = int(model_json["learner"]["learner_model_param"]["num_feature"])
        base_score = float(model_json["learner"]["learner_model_param"]["base_score"])
        learning_rate = json.loads(model_json["learner"]["attributes"]["scikit_learn"])[
            "learning_rate"
        ]
        return n_features, base_score, learning_rate


def _map_json_to_onnx_leaves(json_leaves):
    return [0 if child == -1 else child for child in json_leaves]
