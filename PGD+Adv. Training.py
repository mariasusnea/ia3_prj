import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18

net = resnet18(pretrained=False, num_classes=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def pgd_attack(model, images, labels, epsilon, alpha, iterations):
    model.eval()
    # Create a copy that is a leaf variable
    original_images = images.clone().detach()
    images = images.clone().detach().to(images.device).requires_grad_(True)  
    
    for _ in range(iterations):
        outputs = model(images)
        loss = criterion(outputs, labels)
        grad = torch.autograd.grad(loss, images)[0]

        with torch.no_grad():
            images = images + alpha * torch.sign(grad)
            images = torch.max(torch.min(images, original_images + epsilon), original_images - epsilon) 
            images = torch.clamp(images, -1, 1)
    
    return images.detach()

def adversarial_training(model, training_data, epsilon, alpha, iterations):
    model.train() 
    for images, labels in training_data:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels, epsilon, alpha, iterations)
        optimizer.zero_grad()
        outputs = model(images)
        adv_outputs = model(adv_images)
        loss = criterion(outputs, labels) + criterion(adv_outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, testloader, attack_function=None, attack_params=None):
    model.eval()  
    correct = 0
    total = 0
    device = next(model.parameters()).device  

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  

        if attack_function is not None:
            images = images.clone().detach().to(device).requires_grad_(True)  
            images = attack_function(model, images, labels, *attack_params)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct // total  


num_epochs_initial = 5
for epoch in range(num_epochs_initial):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    scheduler.step()

print('Finished Initial Training')
print("\nTesting initial model:")
accuracy_clean_initial = evaluate(net, testloader)
print(f'Accuracy on clean test images (initial): {accuracy_clean_initial:.2f}%')
accuracy_pgd_initial = evaluate(net, testloader, pgd_attack, [0.03, 2/255, 7])
print(f'Accuracy on PGD perturbed test images (initial): {accuracy_pgd_initial:.2f}%')


num_epochs_adv = 3
epsilon = 0.03
alpha = 2/255
pgd_iterations = 7

for epoch in range(num_epochs_adv):
    print(f"Adversarial training epoch: {epoch+1}")
    adversarial_training(net, trainloader, epsilon, alpha, pgd_iterations)
    scheduler.step()

print('Finished Adversarial Training')
print("\nTesting adversarially trained model:")
accuracy_clean_adv = evaluate(net, testloader)
print(f'Accuracy on clean test images (after adv. training): {accuracy_clean_adv:.2f}%')
accuracy_pgd_adv = evaluate(net, testloader, pgd_attack, [epsilon, alpha, pgd_iterations])
print(f'Accuracy on PGD perturbed test images (after adv. training): {accuracy_pgd_adv:.2f}%')