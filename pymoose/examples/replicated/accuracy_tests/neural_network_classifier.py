from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# from onnxmltools import convert_sklearn
# from onnxmltools.utils import save_model
from skl2onnx.common import data_types as onnx_dtypes

# Convert into ONNX format
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from pymoose import edsl
from pymoose import testing
from pymoose.predictors import neural_net_predictor
from pymoose.predictors import predictor_utils
 
import numpy as np

# Classification data
X, y = make_classification(n_samples=100, random_state=44 ,n_classes=2, n_informative=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

clf = MLPClassifier(activation='logistic', random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict_proba(X_test[:1])
clf.predict(X_test[:5, :])
clf.score(X_test, y_test)

def _build_prediction_logic(onnx_proto):
    predictor = neural_net_predictor.NeuralClassifier.from_onnx(onnx_proto)

    @edsl.computation
    def predictor_no_aes(x: edsl.Argument(predictor.alice, dtype=edsl.float64)):
        with predictor.alice:
            x_fixed = edsl.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
        with predictor.replicated:
            y = predictor.neural_predictor_fn(
                x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE
            )
            y = predictor.post_transform(y, predictor_utils.DEFAULT_FIXED_DTYPE)
        return predictor.handle_output(y, prediction_handler=predictor.bob)

    return predictor, predictor_no_aes
    

initial_type = ("float_input", FloatTensorType([None, clf.n_features_in_]))
onnx_proto = convert_sklearn(clf, initial_types=[initial_type])
# with open('graph.txt', 'w') as f:
#     f.write(str(onnx_proto))

net, net_logic = _build_prediction_logic(onnx_proto)

traced_predictor = edsl.trace(net_logic)
storage = {plc.name: {} for plc in net.host_placements}
runtime = testing.LocalMooseRuntime(storage_mapping=storage)
role_assignment = {plc.name: plc.name for plc in net.host_placements}

result_dict = runtime.evaluate_computation(
    computation=traced_predictor,
    role_assignment=role_assignment,
    arguments={"x": X_test},
)
actual_result = list(result_dict.values())[0]
expected = clf.predict_proba(X_test)
expected_result = np.array(expected)

# print(actual_result - expected_result)
print(np.isclose(actual_result, expected_result, atol=1e-2).all()) # Do outputs match up to 2 decimal points

