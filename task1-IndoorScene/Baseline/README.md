# 310 ML Eng. Challenge - Task 1 Baseline - Indoor Scene Recognition

## *Task specification*:

- Image Classification
- Number of classes: 67
- Metric: balanced accuracy
- Total 15620 images

<br>

---

## *Approach*:
- Framework: PyTorch
- Model architecture used: ResNet-18
- Optimizer: Adam

<br>

---
## *Notebook layout*:
the notebook is structured in XX different sections:

    1 - Retrieveing dataset from source
    2 - Initail preprocessing of images, containing:
        -> Resizing all images
        -> splitting into train/validation partisions by 80/20 ratio
    3 - Train and Test loop function definition
    4 - Main training pipepile function definition
    5 - 


---

### `Function: test(model, test_loader, criterion, device)`

This function takes in a trained PyTorch model, a data loader containing test data, a loss function criterion, and the device on which to run the model. It then evaluates the model's performance on the test set and prints out the average test loss, test accuracy and test balanced accuracy.

    Parameters:
    ----------

        model: [torch.nn.Module] A trained PyTorch model which is to be evaluated on the test data.
        test_loader: [torch.utils.data.DataLoader] A data loader object containing the test data.
        criterion: [torch.nn.CrossEntropyLoss] A PyTorch loss function object used to calculate
                the loss between the model's predictions and the true labels
        device (PyTorch device object) : The device (cpu or gpu) on which the model is to be run.

    -------
    Returns:
    -------
        None. However, the function prints the average test loss, test accuracy and test balanced accuracy.


### `Function: train(model, train_loader, val_loader, criterion, optimizer, epochs, device, chkp_path, writer):`

This function is used to train a PyTorch model on a given dataset using a specified optimizer and loss function. During training, it also logs various metrics such as loss, accuracy, and balanced accuracy of both train and validation sets using TensorBoard. It saves the model checkpoint with the minimum validation loss.



    Parameters:
    -------

        model: [torch.nn.Module] A PyTorch model to be trained.
        train_loader: [torch.utils.data.DataLoader] A PyTorch DataLoader for the training dataset.
        val_loader: [torch.utils.data.DataLoader] A PyTorch DataLoader for the validation dataset.
        criterion: [torch.nn.CrossEntropyLoss] A PyTorch loss function used to compute the loss.
        optimizer: [torch.optim] A PyTorch optimizer used to update the model parameters.
        epochs: [int] An integer representing the number of epochs to train the model.
        device: A PyTorch device (e.g. 'cuda', 'cpu') to be used for training.
        chkp_path: [str] A string representing the file path to save the best checkpoint of the model.
        writer: [torch.utils.tensorboard.SummaryWriter] A TensorBoard SummaryWriter to log the training progress.
    
    -------
    Returns:
    -------
        None
    
    Function Body:
    The function initializes six lists to store the training and validation loss, accuracy, and balanced accuracy over each epoch.
    Then, it iterates over the specified number of epochs and trains the model on the training set.
    After each epoch, it evaluates the model on the validation set, logs the performance metrics using TensorBoard,
    and saves the best checkpoint of the model based on the minimum validation loss.

*How to use:*
```
#initialize the model, data loader, criterion and device
model = MyModel()
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters, lr=lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

epochs = 3

#train and evaluate the model on the test data
train(model, train_loader, val_loader, criterion, optimizer, epochs, device, chkp_path, writer)
test(model, test_loader, criterion, device)
```
