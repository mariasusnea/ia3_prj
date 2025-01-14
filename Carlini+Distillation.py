import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np


class Net(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
subset_size_test = 30
trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(subset_size_test)),
                                          batch_size=1, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
subset_size_train = 30
testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(subset_size_train)),
                                          batch_size=1, shuffle=False)


model = Net()

import torch
import torch.optim as optim
import torch.nn.functional as F

def carlini_wagner_l2_attack(model, images, target_class, c_range=(1e-3, 1e10), search_steps=9, max_iterations=1000, learning_rate=1e-2, initial_const=1e-3, confidence=0):

    model.eval()


    batch_size = images.size(0)
    num_classes = model(images).size(1)


    w = torch.zeros_like(images, requires_grad=True)

    def tanh_rescale(x):
        return 0.5 * (torch.tanh(x) + 1)

    def inverse_tanh_rescale(x):
        return torch.atanh((x - 0.5) * 2 * (1 - 1e-6))

    images_tanh = inverse_tanh_rescale(images)

    def f(x):
        logits = model(x)
        target_logit = logits[:, target_class]
        other_logits = logits + torch.zeros_like(logits)
        other_logits[:, target_class] = float('-inf')
        max_other_logit = torch.max(other_logits, dim=1)[0]
        return torch.clamp(max_other_logit - target_logit + confidence, min=0)

    lower_bound = torch.zeros(batch_size)
    upper_bound = torch.ones(batch_size) * c_range[1]
    c = torch.ones(batch_size) * initial_const

    optimizer = optim.Adam([w], lr=learning_rate)

    for search_step in range(search_steps):
        for iteration in range(max_iterations):
            perturbed_images = tanh_rescale(w + images_tanh)

            l2_loss = F.mse_loss(perturbed_images, images, reduction='sum')

            f_loss = torch.sum(c * f(perturbed_images))

            loss = l2_loss + f_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            perturbed_images = tanh_rescale(w + images_tanh)
            logits = model(perturbed_images)
            preds = torch.argmax(logits, dim=1)
            successful_attack = (preds == target_class).float()

            lower_bound = torch.where(successful_attack == 1, lower_bound, c)
            upper_bound = torch.where(successful_attack == 1, c, upper_bound)
            c = (lower_bound + upper_bound) / 2

    perturbed_images = tanh_rescale(w + images_tanh)
    perturbed_images = torch.clamp(torch.round(perturbed_images * 255) / 255, 0, 1)

    return perturbed_images

import torch

target_class = 0

adversarial_examples = []

for i, (images, labels) in enumerate(trainloader):
    perturbed_images = carlini_wagner_l2_attack(model, images, target_class)

    adversarial_examples.append(perturbed_images)

    if (i + 1) % 100 == 0:
        print(f'Processed {i + 1}/{subset_size_train} images')

adversarial_examples = torch.cat(adversarial_examples)

correct = 0

with torch.no_grad():
    for i, images in enumerate(adversarial_examples):
        images = images.unsqueeze(0) 
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        if predicted.item() == target_class:
            correct += 1

print(f'Adversarial attack success rate: {correct}/{subset_size_train} ({100 * correct / subset_size_train:.2f}%)')

if torch.isnan(adversarial_examples).any() or torch.isinf(adversarial_examples).any():
    print("Warning: Adversarial examples contain NaN or infinite values.")
    adversarial_examples = torch.nan_to_num(adversarial_examples)
adversarial_examples = torch.clamp(adversarial_examples, 0, 1)
adversarial_examples = adversarial_examples.to(torch.float32)

from torchvision.transforms import ToPILImage

to_pil = ToPILImage()

for i, adv_tensor in enumerate(adversarial_examples):
    adv_tensor = adv_tensor.cpu().detach()

    adv_image = to_pil(adv_tensor)


    adv_image.show(title=f'Adversarial Example {i+1}')


    adv_image.save(f'adversarial_example_{i+1}.png')

from torchvision.utils import save_image
for i in range(10):
    image, label =  testset[i]

    save_image(image, f'cifar10_image_{i+1}.png')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


num_epochs = 20
learning_rate = 0.001
temperature = 10.0


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
subset_size_test = 30
train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(subset_size_test)),
                                          batch_size=1, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
subset_size_train = 30
test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(subset_size_train)),
                                          batch_size=1, shuffle=False)


teacher_model = model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)

teacher_model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = teacher_model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

import numpy as np

teacher_model.eval()
soft_labels = []

with torch.no_grad():
    for images, _ in train_loader:
        images = images
        outputs = teacher_model(images) / temperature
        softmax_outputs = torch.softmax(outputs, dim=1)
        soft_labels.extend(softmax_outputs.cpu().numpy())

soft_labels = np.array(soft_labels)

student_model = model
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

student_model.train()
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images
        soft_label_batch = torch.tensor(soft_labels[i*100:(i+1)*100])
        outputs = student_model(images) / temperature
        log_softmax_outputs = torch.log_softmax(outputs, dim=1)
        loss = criterion(log_softmax_outputs, soft_label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

student_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = student_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the student model on the test images: {100 * correct / total:.2f}%')