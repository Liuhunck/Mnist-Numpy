
import Layer

mnist = (
    # Layer.Conv2d(input_channels=1, output_channels=10, kernel_size=5, sigma=np.sqrt(2 / (1 * 5 * 5))),
    # Layer.MaxPool2d(kernel_size=2),
    # Layer.ReLU(),
    # Layer.Conv2d(input_channels=10, output_channels=20, kernel_size=5, sigma=np.sqrt(2 / (10 * 5 * 5))),
    # Layer.MaxPool2d(kernel_size=2),
    # Layer.ReLU(),
    # Layer.Flatten(),
    # Layer.Linear(320, 128, sigma=np.sqrt(1 / 320)),
    # Layer.Sigmoid(),
    # Layer.Linear(128, 64, sigma=np.sqrt(1 / 128)),
    # Layer.Sigmoid(),
    # Layer.Softmax(64, 10, sigma=np.sqrt(1 / 64))
    Layer.Conv2d(input_channels=1, output_channels=10, kernel_size=5),
    Layer.MaxPool2d(kernel_size=2),
    Layer.BatchNorm2d(num_features=10),
    Layer.ReLU(),
    Layer.Conv2d(input_channels=10, output_channels=20, kernel_size=5),
    Layer.MaxPool2d(kernel_size=2),
    Layer.BatchNorm2d(num_features=20),
    Layer.ReLU(),
    Layer.Flatten(),
    Layer.Linear(320, 128),
    Layer.BatchNorm1d(num_features=128),
    Layer.Sigmoid(),
    Layer.Linear(128, 64),
    Layer.BatchNorm1d(num_features=64),
    Layer.Sigmoid(),
    Layer.Softmax(64, 10)
)
