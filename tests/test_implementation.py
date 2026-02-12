import numpy as np

from nn.activations import ReLU, Sigmoid, Softmax
from nn.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from nn.losses import CategoricalCrossEntropyLoss
from nn.optimizers import SGD, Adam


def test_conv2d():
    print("Testing Conv2D...")

    batch_size = 2
    n_channels = 3
    height, width = 8, 8
    n_filters = 4
    kernel_size = 3

    conv = Conv2D(n_filters=n_filters, kernel_size=kernel_size, n_channels=n_channels, stride=1, padding=1)
    X = np.random.randn(batch_size, n_channels, height, width)

    conv.forward(X)
    assert conv.output.shape == (batch_size, n_filters, height, width), f"Expected shape {(batch_size, n_filters, height, width)}, got {conv.output.shape}"

    dout = np.random.randn(*conv.output.shape)
    conv.backward(dout)
    assert conv.dinputs.shape == X.shape, f"Expected gradient shape {X.shape}, got {conv.dinputs.shape}"
    assert conv.dW.shape == conv.W.shape, f"Expected weight gradient shape {conv.W.shape}, got {conv.dW.shape}"

    print("Conv2D test passed")


def test_maxpool2d():
    print("Testing MaxPool2D...")

    batch_size = 2
    n_channels = 3
    height, width = 8, 8

    pool = MaxPool2D(pool_size=2, stride=2)
    X = np.random.randn(batch_size, n_channels, height, width)

    pool.forward(X)
    expected_shape = (batch_size, n_channels, height//2, width//2)
    assert pool.output.shape == expected_shape, f"Expected shape {expected_shape}, got {pool.output.shape}"

    dout = np.random.randn(*pool.output.shape)
    pool.backward(dout)
    assert pool.dinputs.shape == X.shape, f"Expected gradient shape {X.shape}, got {pool.dinputs.shape}"

    print("MaxPool2D test passed")


def test_flatten():
    print("Testing Flatten...")

    batch_size = 2
    n_channels = 3
    height, width = 4, 4

    flatten = Flatten()
    X = np.random.randn(batch_size, n_channels, height, width)

    flatten.forward(X)
    expected_shape = (batch_size, n_channels * height * width)
    assert flatten.output.shape == expected_shape, f"Expected shape {expected_shape}, got {flatten.output.shape}"

    dout = np.random.randn(*flatten.output.shape)
    flatten.backward(dout)
    assert flatten.dinputs.shape == X.shape, f"Expected gradient shape {X.shape}, got {flatten.dinputs.shape}"

    print("Flatten test passed")


def test_dense():
    print("Testing Dense...")

    batch_size = 2
    n_inputs = 16
    n_neurons = 8

    dense = Dense(n_inputs, n_neurons)
    X = np.random.randn(batch_size, n_inputs)

    dense.forward(X)
    assert dense.output.shape == (batch_size, n_neurons), f"Expected shape {(batch_size, n_neurons)}, got {dense.output.shape}"

    dout = np.random.randn(*dense.output.shape)
    dense.backward(dout)
    assert dense.dinputs.shape == X.shape, f"Expected gradient shape {X.shape}, got {dense.dinputs.shape}"
    assert dense.dW.shape == dense.W.shape, f"Expected weight gradient shape {dense.W.shape}, got {dense.dW.shape}"

    print("Dense test passed")


def test_relu():
    print("Testing ReLU...")

    relu = ReLU()
    X = np.array([[-1, 0, 1, 2], [-2, -1, 0, 1]])

    relu.forward(X)
    expected = np.array([[0, 0, 1, 2], [0, 0, 0, 1]])
    assert np.allclose(relu.output, expected), "ReLU forward failed"

    dout = np.ones_like(X)
    relu.backward(dout)
    expected_grad = np.array([[0, 0, 1, 1], [0, 0, 0, 1]])
    assert np.allclose(relu.dinputs, expected_grad), "ReLU backward failed"

    print("ReLU test passed")


def test_sigmoid():
    print("Testing Sigmoid...")

    sigmoid = Sigmoid()
    X = np.array([[-2, -1, 0, 1, 2], [-1, 0, 1, 2, 3]])

    sigmoid.forward(X)
    assert np.all(sigmoid.output > 0) and np.all(sigmoid.output < 1), "Sigmoid outputs should be between 0 and 1"
    assert np.isclose(sigmoid.output[0, 2], 0.5, atol=1e-6), "Sigmoid(0) should be 0.5"
    assert sigmoid.output[0, 0] < sigmoid.output[0, 4], "Sigmoid should be monotonically increasing"

    dout = np.ones_like(X)
    sigmoid.backward(dout)

    assert sigmoid.dinputs.shape == X.shape, f"Expected gradient shape {X.shape}, got {sigmoid.dinputs.shape}"
    assert np.isclose(sigmoid.dinputs[0, 2], 0.25, atol=1e-6), "Sigmoid gradient at 0 should be 0.25"

    print("Sigmoid test passed")


def test_softmax():
    print("Testing Softmax...")

    softmax = Softmax()
    X = np.array([[1, 2, 3], [1, 2, 3]])

    softmax.forward(X)
    assert np.allclose(np.sum(softmax.output, axis=1), 1.0), "Softmax outputs should sum to 1"
    assert np.all(softmax.output > 0), "Softmax outputs should be positive"

    print("Softmax test passed")


def test_cross_entropy_loss():
    print("Testing CrossEntropyLoss...")

    criterion = CategoricalCrossEntropyLoss()

    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])
    targets = np.array([0, 1])

    loss = criterion.forward(logits, targets)
    assert isinstance(loss, (float, np.floating)), "Loss should be a scalar"
    assert loss > 0, "Loss should be positive"

    criterion.backward()
    assert criterion.dinputs.shape == logits.shape, "Gradient shape should match input shape"

    print(f"CrossEntropyLoss test passed (loss: {loss:.4f})")


def test_sgd_optimizer():
    print("Testing SGD optimizer...")

    optimizer = SGD(learning_rate=0.1, momentum=0.9)

    layer = Dense(4, 2)
    initial_W = layer.W.copy()

    layer.dW = np.ones_like(layer.W) * 0.1
    layer.db = np.ones_like(layer.b) * 0.1

    optimizer.update_params(layer)
    optimizer.post_update_params()

    assert not np.allclose(layer.W, initial_W), "Weights should be updated"

    print("SGD optimizer test passed")


def test_adam_optimizer():
    print("Testing Adam optimizer...")

    optimizer = Adam(learning_rate=0.001)

    layer = Dense(4, 2)
    initial_W = layer.W.copy()

    layer.dW = np.ones_like(layer.W) * 0.1
    layer.db = np.ones_like(layer.b) * 0.1

    optimizer.update_params(layer)
    optimizer.post_update_params()

    assert not np.allclose(layer.W, initial_W), "Weights should be updated"

    print("Adam optimizer test passed")


def test_dropout():
    print("Testing Dropout...")

    dropout = Dropout(rate=0.5)
    X = np.ones((2, 4))

    dropout.forward(X, training=True)
    assert dropout.output.shape == X.shape, "Output shape should match input"
    assert not np.allclose(dropout.output, X), "Dropout should modify values in training mode"

    dropout.forward(X, training=False)
    assert np.allclose(dropout.output, X), "Dropout should not modify values in inference mode"

    print("Dropout test passed")


def run_all_tests():
    print("Running CNN Implementation Tests")
    print("-"*50)

    try:
        test_conv2d()
        test_maxpool2d()
        test_flatten()
        test_dense()
        test_relu()
        test_sigmoid()
        test_softmax()
        test_cross_entropy_loss()
        test_sgd_optimizer()
        test_adam_optimizer()
        test_dropout()

        print("\nALL TESTS PASSED!")
        print("-"*50)
        print("The CNN implementation is working correctly!")

    except AssertionError as e:
        print(f"\n Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
