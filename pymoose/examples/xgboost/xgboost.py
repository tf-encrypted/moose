# import argparse
# import logging

# import numpy as np

# from pymoose import edsl
# from pymoose import elk_compiler
# from pymoose.computation import utils
# from pymoose.computation.standard import TensorType
# from pymoose.logger import get_logger
# from pymoose.testing import LocalMooseRuntime


# def edsl_index(x, i, axis):
#     return x[:, i]


# def edsl_constant(val):
#     print("constant", val)
#     return val


# def edsl_less(x, y):
#     res = x < y
#     print("less", x, y, res)
#     return res


# def edsl_mux(selector, x, y):
#     res = y + selector * (x - y)
#     print("mux", selector, x, y, res)
#     return res


# def edsl_addn(*xs):
#     return sum(xs)


# def map_tree(tree, nb_features, learning_rate, x):
#     base_weights = tree["base_weights"]
#     left = tree["left_children"]
#     right = tree["right_children"]
#     parents = tree["parents"]
#     split_conditions = tree["split_conditions"]
#     split_indices = tree["split_indices"]

#     print("base weights", base_weights)
#     print("left children", left)
#     print("right_children", right)
#     print("parents", parents)
#     print("split_condition", split_conditions)
#     print("split_indices", split_indices)

#     features = [edsl_index(x, i, axis=1) for i in range(nb_features)]

#     # Adjust weights based on learning rate
#     base_weights = [learning_rate * w for w in base_weights]

#     def create_computation(node):
#         print("\nnode", node)
#         left_child = left[node]
#         right_child = right[node]
#         if left_child != -1 and right_child != -1:
#             # we're at an inner node; this is the recursive case
#             selector = edsl_less(features[split_indices[node]],
#                 split_conditions[node])
#             return edsl_mux(
#                 selector,
#                 create_computation(left_child),
#                 create_computation(right_child),
#             )
#         else:
#             assert left_child == -1
#             assert right_child == -1
#             # we're at a left node; this is the base case
#             return edsl_constant(base_weights[node])

#     return create_computation(0)


# def map_forest(model, x):
#     base_score = float(model_json["learner"]["learner_model_param"]["base_score"])
#     nb_features = int(model_json["learner"]["learner_model_param"]["num_feature"])
#     learning_rate = json.loads(model_json["learner"]["attributes"]["scikit_learn"])[
#         "learning_rate"
#     ]
#     print("Learning Rate", learning_rate)

#     results = [
#         map_tree(tree, nb_features, learning_rate, x)
#         for tree in model["learner"]["gradient_booster"]["model"]["trees"]
#     ]
#     print("tree results", results)

#     return edsl_addn(*results) + edsl_constant(base_score)
