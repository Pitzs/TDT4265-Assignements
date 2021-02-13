import utils
import os
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyper-parameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # Task 4 point a and b
    # Here is presented the model retrieved from the previous task3
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    print(f'Training the Original model {neurons_per_layer[0]} neurons in hidden layer')
    train_history, val_history = trainer.train(num_epochs)

    # Task 4
    # TASK 4 - a) 32 neurons for hidden layer
    neurons_per_layer = [32, 10]
    model_32neu = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_32neu = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_32neu, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    print(f'Training the Original model {neurons_per_layer[0]} neurons in hidden layer')
    train_history_32neu, val_history_32neu = trainer_32neu.train(num_epochs)
    # TASK 4 - b) 128 neurons for hidden layer
    neurons_per_layer = [128, 10]
    model_128neu = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_128neu = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_128neu, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    print(f'Training the Original model {neurons_per_layer[0]} neurons in hidden layer')
    train_history_128neu, val_history_128neu = trainer_128neu.train(num_epochs)
    # Plot results
    plt.subplot(1, 2, 1)

    utils.plot_loss(train_history["loss"],
                    "train Loss - Original ", npoints_to_average=10)
    utils.plot_loss(
        train_history_32neu["loss"], "train Loss - 32 neurons", npoints_to_average=10)
    utils.plot_loss(
        train_history_128neu["loss"], "train Loss - 128 neurons", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1])
    utils.plot_loss(val_history["accuracy"], "Val Accuracy - Original")
    utils.plot_loss(train_history["accuracy"], "Train Accuracy - Original")
    utils.plot_loss(val_history_32neu["accuracy"], "Val Accuracy - 32")
    utils.plot_loss(train_history_32neu["accuracy"], "Train Accuracy - 32")
    utils.plot_loss(val_history_128neu["accuracy"], "Val Accuracy - 128")
    utils.plot_loss(train_history_128neu["accuracy"], "Train Accuracy - 128")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()