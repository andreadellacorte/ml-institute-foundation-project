import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_and_save_pytorch_model():
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleNN()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

    # Save the model
    torch.save(model.state_dict(), "mnist_pytorch_model.pth")

# Modify the test_model_on_mnist function to save misclassified images
def test_model_on_mnist():
    # Load the MNIST test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the trained model
    model = SimpleNN()
    model.load_state_dict(torch.load("mnist_pytorch_model.pth"))
    model.eval()

    # Test the model
    correct = 0
    total = 0
    misclassified_count = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save misclassified images for debugging
            if predicted != labels and misclassified_count < 5:  # Limit to 5 images
                misclassified_count += 1
                image = images[0].squeeze().numpy()  # Convert to numpy array
                plt.imshow(image, cmap="gray")
                plt.title(f"True: {labels.item()}, Predicted: {predicted.item()}")
                plt.axis("off")
                plt.savefig(f"misclassified_{idx}.png")

    print(f"Accuracy on MNIST test set: {100 * correct / total}%")

if __name__ == "__main__":
    # train_and_save_pytorch_model()
    test_model_on_mnist()
