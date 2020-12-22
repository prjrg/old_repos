import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models,transforms

batch_size = 50
train_data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_data_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=2)

val_data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_data_transform)
val_order = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, loss_function, optimizer, data_loader):
    model.train()
    current_loss = 0.0
    current_acc = 0

    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def test_model(model, loss_function, data_loader):
    model.eval()

    current_loss = 0.0
    current_acc = 0

    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def tl_feature_extractor(epochs=5):
    model = torchvision.models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.fc.parameters())

    for epoch in range(epochs):
        print('Epoch {}/[}'.format(epochs + 1, epochs))
        train_model(model, loss_function, optimizer, train_loader)
        test_model(model, loss_function, val_order)


def tl_fine_tuning(epochs=5):
    model = models.resnet18(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        train_model(model, loss_function, optimizer, train_loader)
        test_model(model, loss_function, val_order)


tl_fine_tuning()
