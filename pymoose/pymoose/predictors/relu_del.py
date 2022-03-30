import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch.onnx
import torch
import torch.nn.functional as F
import torch.nn as nn
import pathlib
import itertools
import pathlib

import numpy as np
import onnx
from absl.testing import parameterized

import pymoose
from pymoose import edsl
from pymoose import elk_compiler
from pymoose import testing
from pymoose.computation import utils as comp_utils
from pymoose.predictors import neural_network_predictor
from pymoose.predictors import predictor_utils

# Model architecture 3 - PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output

model = Net()
model.eval()

dummy_input = torch.tensor([[ 0.4595, -0.8661,  1.7674,  1.9377,  0.3077, -0.8155,  0.3508,  0.2848,
         -1.8987,  0.3189]])

input_names = [ "net_input" ]
output_names = [ "net_output" ]

expected = model.forward(dummy_input).detach()
expected_result = np.array(expected)

torch.onnx.export(model, dummy_input, "pytorch_relu.onnx")
onnx_proto = onnx.load("pytorch_relu.onnx")


def _build_prediction_logic(onnx_proto):
    predictor = neural_network_predictor.NeuralNetwork.from_onnx(onnx_proto)

    @edsl.computation
    def predictor_no_aes(x: edsl.Argument(predictor.alice, dtype=edsl.float64)):
        with predictor.alice:
            x_fixed = edsl.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
        with predictor.replicated:
            y = predictor.neural_predictor_fn(
                x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE
            )
        return predictor.handle_output(y, prediction_handler=predictor.bob)

    return predictor, predictor_no_aes


net, net_logic = _build_prediction_logic(onnx_proto)

traced_predictor = edsl.trace(net_logic)
storage = {plc.name: {} for plc in net.host_placements}
runtime = testing.LocalMooseRuntime(storage_mapping=storage)
role_assignment = {plc.name: plc.name for plc in net.host_placements}


x_input = dummy_input.type(torch.double).numpy()

result_dict = runtime.evaluate_computation(
    computation=traced_predictor,
    role_assignment=role_assignment,
    arguments={"x": x_input},
)
actual_result = list(result_dict.values())[0]

print(expected_result)
print(actual_result)

print(
    np.isclose(actual_result, expected_result, atol=1e-2).all()
)  # Do outputs match up to 2 decimal points