import utils
import os
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created in assignment text - Comparing with and without shuffling.

    # FIRST CASE (weights)
    use_improved_weight_init = True

    model_weights = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_weights = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_weights, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_weights, val_history_weights = trainer_weights.train(
        num_epochs)

    # SECOND CASE (init + sig)
    use_improved_sigmoid = True

    model_sig = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_sig = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_sig, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_sig, val_history_sig = trainer_sig.train(
        num_epochs)

    # THIRD CASE (init + sig + momentum)
    use_momentum = True
    learning_rate = .02
    momentum_gamma = .9

    model_mom = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_mom = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_mom, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_mom, val_history_mom = trainer_mom.train(
        num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 3 Model ", npoints_to_average=10)
    utils.plot_loss(
        train_history_weights["loss"], "Task 3 Model - WI", npoints_to_average=10)
    utils.plot_loss(
        train_history_sig["loss"], "Task 3 Model - WI + S", npoints_to_average=10)
    utils.plot_loss(
        train_history_mom["loss"], "Task 3 Model - WI + S + M", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1])
    utils.plot_loss(val_history["accuracy"], "Task 3 Model")
    utils.plot_loss(
        val_history_weights["accuracy"], "Task 3 Model - WI")
    utils.plot_loss(
        val_history_sig["accuracy"], "Task 3 Model - WI + S")
    utils.plot_loss(
        val_history_mom["accuracy"], "Task 3 Model - WI + S + M")
    plt.ylabel("Validation Accuracy")
    plt.legend()

    path = r'C:\Users\aless\Desktop\Università\Magistrale - Biomedical Engineering - Polimi\Z DEEP LEARNING AND COMPUTER VISION\TDT4265-Assignements\assignment2'
    filename = 'task3_train_loss.png'
    fig_task = os.path.join(path, filename)
    plt.savefig(fig_task)
