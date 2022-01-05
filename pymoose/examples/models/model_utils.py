from pymoose import edsl

DEFAULT_FIXED_DTYPE = edsl.fixed(46, 40)


def handle_aes_predictor_input(aes_key, aes_data, decryptor):
    assert isinstance(aes_data.vtype, edsl.AesTensorType)
    assert aes_data.vtype.dtype.is_fixedpoint
    assert isinstance(aes_key.vtype, edsl.AesKeyType)

    with decryptor:
        aes_inputs = edsl.decrypt(aes_key, aes_data)

    return aes_inputs


def handle_predictor_output(prediction, prediction_handler, output_dtype=edsl.float64):
    with prediction_handler:
        result = edsl.cast(prediction, dtype=output_dtype)

    return result
