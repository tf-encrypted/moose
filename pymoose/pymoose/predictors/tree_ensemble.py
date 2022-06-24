import abc
import json

import pymoose as pm
from pymoose.predictors import predictor
from pymoose.predictors import predictor_utils as utils


class DecisionTreeRegressor(predictor.Predictor):
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

    def aes_predictor_factory(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} is not meant to be used directly as an "
            "AesPredictor model. Consider expressing your decision tree as a tree "
            "ensemble with another AesPredictor implementation."
        )

    def __call__(self, x, n_features, rescale_factor, fixedpoint_dtype):
        leaf_weights = {ix: rescale_factor * w for ix, w in self.weights.items()}
        features_vec = [pm.index_axis(x, axis=1, index=i) for i in range(n_features)]
        return self._traverse_tree(0, leaf_weights, features_vec, fixedpoint_dtype)

    def _traverse_tree(self, node, leaf_weights, x_features, fixedpoint_dtype):
        left_child = self.left[node]
        right_child = self.right[node]
        if left_child != 0 and right_child != 0:
            # we're at an inner node; this is the recursive case
            selector = pm.less(
                x_features[self.split_indices[node]],
                self.fixedpoint_constant(
                    self.split_conditions[node], self.mirrored, dtype=fixedpoint_dtype
                ),
            )

            return pm.mux(
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


class TreeEnsemble(predictor.Predictor, metaclass=abc.ABCMeta):
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

    def predictor_fn(self, x, fixedpoint_dtype):
        forest_scores = [
            tree(
                x,
                self.n_features,
                rescale_factor=self.learning_rate,
                fixedpoint_dtype=fixedpoint_dtype,
            )
            for tree in self.trees
        ]
        # if any of the trees are degenerate, they will return a non-replicated value.
        # we want post_transform to expect a collection of replicated values, since its
        # variadic ops will not necessarily know to move their results from source
        # placements to replicated placement.
        # it's a bit ugly, but it works for now.
        return list(map(pm.identity, forest_scores))

    def __call__(self, x, fixedpoint_dtype=utils.DEFAULT_FIXED_DTYPE):
        tree_scores = self.predictor_fn(x, fixedpoint_dtype=fixedpoint_dtype)
        return self.post_transform(tree_scores, fixedpoint_dtype=fixedpoint_dtype)


class TreeEnsembleClassifier(TreeEnsemble):
    def __init__(
        self,
        trees,
        n_features,
        n_classes,
        base_score,
        learning_rate,
        transform_output,
        tree_class_map,
    ):
        super().__init__(trees, n_features, base_score, learning_rate)
        self.n_classes = n_classes
        self.tree_class_map = tree_class_map
        self.transform_output = transform_output

    @classmethod
    def from_onnx(cls, model_proto):
        (
            forest_node,
            (nodes_treeids, left, right, split_conditions, split_indices),
            n_trees,
            n_features,
            base_score,
            learning_rate,
        ) = _onnx_base(model_proto, "TreeEnsembleClassifier")

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

        post_transform_attr = utils.find_attribute_in_node(
            forest_node, "post_transform"
        )
        post_transform = post_transform_attr.s.decode()

        if post_transform == "NONE" and n_classes > 2:
            # in this case, sklearn's ONNX file stores nodes differently;
            # each leaf & inner node has array of length n_classes instead of
            #   having n_trees * n_classes separate trees, whereas other ONNX
            #   files have separate trees per class.
            # in TreeEnsembleClassifier, we currently always represent with a separate
            # forest per class, so here we need to duplicate some trees for that
            # representation.
            final_class_treeids = [
                class_id + tree_id * n_classes
                for (tree_id, class_id) in zip(class_treeids, class_ids)
            ]
            # update n_trees inferred by onnx helper fn above
            n_trees = len(set(final_class_treeids))
            # rely on nodes_treeids being sorted to preserve sublist order.
            # the order matters to map back into the format expected when there are
            # separate forests for each class
            assert nodes_treeids == sorted(nodes_treeids)
            sublists = [
                list(filter(lambda x: x == i, nodes_treeids))
                for i in sorted(set(nodes_treeids))
            ]
            repeated_sublists = [
                [n_classes * i + j for _ in x]
                for j in range(n_classes)
                for i, x in enumerate(sublists)
            ]
            final_nodes_treeids = [x for y in repeated_sublists for x in y]
        else:
            final_class_treeids = class_treeids
            final_nodes_treeids = nodes_treeids

        tree_args = [
            {
                "weights": {},
                "children": [[], []],
                "split_indices": [],
                "split_conditions": [],
            }
            for _ in range(n_trees)
        ]

        for i, tree_id in enumerate(final_nodes_treeids):
            # i % len(_) duplicates nodes from the same ONNX trees in cases when
            # final_nodes_treeids is longer than the lists of nodes coming from ONNX
            # this is only the case when there are not n_trees * n_classes distinct
            # trees in the ONNX file
            tree_args[tree_id]["children"][0].append(left[i % len(left)])
            tree_args[tree_id]["children"][1].append(right[i % len(right)])
            tree_args[tree_id]["split_indices"].append(
                split_indices[i % len(split_indices)]
            )
            tree_args[tree_id]["split_conditions"].append(
                split_conditions[i % len(split_conditions)]
            )

        for i, class_weight in enumerate(class_weights):
            tree_args[final_class_treeids[i]]["weights"][
                class_nodeids[i]
            ] = class_weight

        trees = [DecisionTreeRegressor(**kwargs) for kwargs in tree_args]
        tree_class_map = {
            tree_id: class_id
            for tree_id, class_id in zip(final_class_treeids, class_ids)
        }

        transform_output = post_transform != "NONE"

        return cls(
            trees,
            n_features,
            n_classes,
            base_score,
            learning_rate,
            transform_output,
            tree_class_map,
        )

    def post_transform(self, tree_scores, fixedpoint_dtype):
        if self.n_classes == 2:
            return self._maybe_sigmoid(tree_scores, fixedpoint_dtype)
        else:
            logit = self._ovr_logit(
                tree_scores, axis=1, fixedpoint_dtype=fixedpoint_dtype
            )
            if self.transform_output:
                return pm.softmax(logit, axis=1, upmost_index=self.n_classes)
            return logit

    def _maybe_sigmoid(self, tree_scores, fixedpoint_dtype):
        base_score = self.fixedpoint_constant(
            self.base_score, self.carole, dtype=fixedpoint_dtype
        )
        logit = pm.add(pm.add_n(tree_scores), base_score)
        pos_prob = pm.sigmoid(logit) if self.transform_output else logit
        pos_prob = pm.expand_dims(pos_prob, axis=1)
        one = self.fixedpoint_constant(1, plc=self.mirrored, dtype=fixedpoint_dtype)
        neg_prob = pm.sub(one, pos_prob)
        return pm.concatenate([neg_prob, pos_prob], axis=1)

    def _ovr_logit(self, tree_scores, axis, fixedpoint_dtype):
        ovr_results = [[] for _ in range(self.n_classes)]
        for tree_ix, model_ix in self.tree_class_map.items():
            ovr_results[model_ix].append(tree_scores[tree_ix])
        base_score = self.fixedpoint_constant(
            self.base_score, self.carole, dtype=fixedpoint_dtype
        )
        ovr_logits = [pm.add(pm.add_n(ovr), base_score) for ovr in ovr_results]
        reformed_logits = pm.concatenate(
            [pm.expand_dims(ovr, axis=axis) for ovr in ovr_logits], axis=axis
        )
        return reformed_logits


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
        ) = _onnx_base(model_proto, "TreeEnsembleRegressor")

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
        penultimate_score = pm.add_n(tree_scores)
        return pm.add(base_score, penultimate_score)

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


def _onnx_base(model_proto, forest_node_name):
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

    split_conditions_attr = utils.find_attribute_in_node(forest_node, "nodes_values")
    assert split_conditions_attr.type == 6
    split_conditions = split_conditions_attr.floats

    split_indices_attr = utils.find_attribute_in_node(forest_node, "nodes_featureids")
    assert split_indices_attr.type == 7
    split_indices = split_indices_attr.ints

    tree_args = (nodes_treeids, left, right, split_conditions, split_indices)

    n_trees = len(set(nodes_treeids))

    # `n_features` arg
    model_input = model_proto.graph.input[0]
    input_shape = utils.find_input_shape(model_input)
    assert len(input_shape) == 2
    n_features = input_shape[1].dim_value

    n_split_indices = len(set(split_indices))
    largest_split_indices = max(split_indices)

    if n_split_indices > n_features or largest_split_indices > n_features:
        raise ValueError(
            f"In the ONNX file, the input shape has {n_features} "
            f"features and there are {n_split_indices} distinct split indices . "
            f"with the largest index {largest_split_indices}. Validate you "
            "set correctly the `initial_types` when converting your model to ONNX."
        )

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
