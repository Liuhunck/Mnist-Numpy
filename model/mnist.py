
import Layer

mnist = (
    Layer.Conv2d(input_channels=1, output_channels=10, kernel_size=5),
    Layer.MaxPool2d(kernel_size=2),
    Layer.ReLU(),
    Layer.Conv2d(input_channels=10, output_channels=20, kernel_size=5),
    Layer.MaxPool2d(kernel_size=2),
    Layer.ReLU(),
    Layer.Flatten(),
    Layer.Linear(320, 128),
    Layer.Sigmoid(),
    Layer.Linear(128, 64),
    Layer.Sigmoid(),
    Layer.Softmax(64, 10)
)
