import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
import librosa as lr
import noisereduce as nr
import moviepy.editor as mp
from scipy.io import wavfile
import librosa.display as ld
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import *
from tqdm.auto import tqdm
import torch
from timeit import default_timer as Timer

def build_spectogram(audio_path, plot_path, bar = False):
    """
    Build spectrograms from audio files and save them as PNG images.

    Parameters:
        audio_path (str): The path to the directory containing the audio files.
        plot_path (str): The path to the directory where the spectrogram images will be saved.
        bar (bool, optional): Whether to include a colorbar in the spectrogram images. Default is False.

    Returns:
        None

    Raises:
        OSError: If there is an error while creating directories or loading audio files.

    """
    folders = []
    for item in os.listdir(audio_path):
        item_path = os.path.join(audio_path, item)
        if os.path.isdir(item_path):
            folders.append(item)

    for folder in folders:
        item_list = os.listdir(audio_path + folder)
        os.makedirs(plot_path+'/'+folder)
        for item in item_list:
            music, rate = lr.load(audio_path+folder+'/'+item)
            stft = lr.feature.melspectrogram(y=music, sr=rate, n_mels=256)
            db = lr.amplitude_to_db(stft)
            fig, ax = plt.subplots()
            img = ld.specshow(db, x_axis='time', y_axis='log', ax=ax)
            plt.axis(False)
            if bar == True:
                fig.colorbar(img, ax=ax, format='%0.2f')
            a = item.replace('.wav', '.png')
            plt.savefig(plot_path+'/'+folder+'/'+a)

def performance(model, x_test, y_test):
    """
    Calculates and displays the performance metrics of a trained model.

    Parameters:
    -----------
    model : object
        The trained machine learning model.

    x_test : array-like of shape (n_samples, n_features)
        The input test data.

    y_test : array-like of shape (n_samples,)
        The target test data.

    Returns:
    --------
    None

    Prints:
    -------
    Model Performance:
        Classification report containing precision, recall, F1-score, and support for each class.
    Accuracy:
        The accuracy of the model on the test data.
    Confusion Matrix:
        A plot of the confusion matrix, showing the true and predicted labels for the test data.

    Example:
    --------
    >>> performance(model, x_test, y_test)
    """

    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    print("                 Model Performance")
    print(report)
    print(f"Accuracy = {round(accuracy*100, 2)}%")
    matrix = confusion_matrix(y_test, preds)
    matrix_disp = ConfusionMatrixDisplay(matrix)
    matrix_disp.plot(cmap='Blues')
    plt.show()
    
class CustomDataset_CSVlabels(Dataset):
    """
    A PyTorch dataset for loading spectrogram images and their corresponding labels from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the image file names and labels.
        img_dir (str): Root directory where the image files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop`` for randomly cropping an image.

    Attributes:
        img_labels (DataFrame): A pandas dataframe containing the image file names and labels.
        img_dir (str): Root directory where the image files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop`` for randomly cropping an image.
    
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the image and label at the given index.

    Returns:
        A PyTorch dataset object that can be passed to a DataLoader for batch processing.
    """
    def __init__(self,csv_file, img_dir, transform=None) -> None:
        super().__init__()
        self.img_labels = pd.read_csv(csv_file)
        self.img_labels.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.img_labels)
    
    def __getitem__(self, index):
        """
        Returns the image and label at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        image = Image.open(img_path)
        image = image.convert("RGB")
        y_label = torch.tensor(int(self.img_labels.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

def Train_Loop(
        num_epochs:int,
        train_dataloader:torch.utils.data.DataLoader,
        test_dataloader:torch.utils.data.DataLoader,
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        loss_function:torch.nn.Module
):
    """
    Trains a PyTorch model using the given train and test dataloaders for the specified number of epochs.

    Parameters:
    -----------
    num_epochs : int
        The number of epochs to train the model for.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for the training data.
    test_dataloader : torch.utils.data.DataLoader
        The dataloader for the test/validation data.
    model : torch.nn.Module
        The PyTorch model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer to be used during training.
    loss_function : torch.nn.Module
        The loss function to be used during training.

    Returns:
    --------
    None

    Raises:
    -------
    None

    Notes:
    ------
    This function loops over the specified number of epochs and for each epoch, it trains the model on the training
    data and evaluates the performance on the test/validation data. During each epoch, it prints the training loss
    and the test loss and accuracy. At the end of training, it prints the total time taken for training.
    """
    model.to('cuda')
    start_time = Timer()
    
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch: {epoch}\n-----------")
        train_loss = 0
        for batch, (x,y) in enumerate(train_dataloader):
            x,y = x.to('cuda'), y.to('cuda')
            y=y.float()
            model.train()
            y_logits = model(x).squeeze()
            y_pred = torch.round(y_logits)
            loss = loss_function(y_logits, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                print(f"Looked at {batch * len(x)}/{len(train_dataloader.dataset)} samples")
                print(y_logits)

        train_loss /= len(train_dataloader)
        
        test_loss, test_acc = 0, 0 
        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X,y = X.to('cuda'), y.to('cuda')
                y = y.float()
                test_logits = model(X).squeeze()
                test_pred = torch.round(test_logits)
                test_loss += loss_function(test_logits, y)
                test_acc += accuracy_score(y_true=y.cpu(), y_pred=test_pred.cpu())
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc*100:.2f}%\n")

    end_time = Timer()
    print(f"Time taken = {end_time-start_time}")

class CustomDataset_FolderLabels:
    """
    CustomDataset class for loading and splitting a dataset into training, validation, and testing sets.

    Args:
        data_path (str): Path to the main folder containing subfolders for each class.
        train_ratio (float): Ratio of data allocated for the training set (0.0 to 1.0).
        val_ratio (float): Ratio of data allocated for the validation set (0.0 to 1.0).
        test_ratio (float): Ratio of data allocated for the testing set (0.0 to 1.0).
        batch_size (int): Number of samples per batch in the data loaders.
        transform (torchvision.transforms.transforms.Compose): Transformations to be applied on the image

    Attributes:
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        test_loader (torch.utils.data.DataLoader): Data loader for the testing set.

    """
    def __init__(self, data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32, transform=None):
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset and splits it into training, validation, and testing sets.

        """
        dataset = ImageFolder(root=self.data_path, transform=self.transform)
        num_samples = len(dataset)

        train_size = int(self.train_ratio * num_samples)
        val_size = int(self.val_ratio * num_samples)
        test_size = num_samples - train_size - val_size

        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_size, val_size, test_size])

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def get_train_loader(self):
        """
        Get the data loader for the training set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the training set.

        """
        return self.train_loader

    def get_val_loader(self):
        """
        Get the data loader for the validation set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the validation set.

        """
        return self.val_loader

    def get_test_loader(self):
        """
        Get the data loader for the testing set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the testing set.

        """
        return self.test_loader
