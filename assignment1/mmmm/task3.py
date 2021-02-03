import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    # First computation of the prediction
    outputs = model.forward(X)

    # Convert the prediction into 0 and 1 by setting as 1 the highest value in the 10 outputs, the rest will be 0.
    accuracy = np.sum(outputs.argmax(1) == targets.argmax(1))/targets.shape[0]
    
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        # Forward step (retrieving the predictions)
        outputs = self.model.forward(X_batch)

        # Backward step
        self.model.backward(X_batch, outputs, Y_batch)

        # Updating the weights
        self.model.w -= self.model.grad*self.learning_rate

        # Computing the loss
        loss = cross_entropy_loss(Y_batch, outputs)
        
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Initialize the model
    model_a = SoftmaxModel(X_train.shape[1], Y_train.shape[1], l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model_a, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model_a.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model_a.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model_a))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model_a))

    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model_b = SoftmaxModel(X_train.shape[1], Y_train.shape[1], l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model_b, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    trainer.train(num_epochs)

    # You can finish the rest of task 4 below this point.

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    lengths = []

    for i in l2_lambdas:
        model_c = SoftmaxModel(X_train.shape[1], Y_train.shape[1], l2_reg_lambda=i)
        trainer = SoftmaxTrainer(
            model_c, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val)
        _, val_history = trainer.train(num_epochs)
        lengths.append((i, np.linalg.norm(model_c.w)))

        # Plot accuracy
        utils.plot_loss(val_history["accuracy"], "lambda = {}".format(i))

    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.ylim([0.75, .93])
    plt.title("Validation accuracy of different models with regularizations")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4e - Plotting of the l2 norm for each weight
    plt.figure(figsize=(3, 3))
    plt.semilogx(*zip(*lengths), 'o')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('weight norm')
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()

    # Plotting the softmax weights (Task 4b)
    matr_a = model_a.w[:-1, 0].reshape((28, 28))

    for i in range(1, 10):
        matr_a = np.concatenate((matr_a, model_a.w[:-1, i].reshape((28, 28))), axis=1)

    matr_b = model_b.w[:-1, 0].reshape((28, 28))

    for i in range(1, 10):
        matr_b = np.concatenate((matr_b, model_b.w[:-1, i].reshape((28, 28))), axis=1)

    plt.subplot(211)
    plt.imshow(matr_a)
    plt.title('Weights No Regularization')
    plt.subplot(212)
    plt.imshow(matr_b)
    plt.title('Weights Regularization (lambdas 1)')
    plt.savefig("task4b_softmax_weight_wo_reg.png")
    plt.show()