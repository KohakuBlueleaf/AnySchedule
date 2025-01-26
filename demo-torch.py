import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from tqdm import tqdm

from anyschedule import AnySchedule


class Net(nn.Module):
    """
    Simple LeNet with Mish and LayerNorm
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.norm1 = nn.GroupNorm(2, 6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = nn.LayerNorm(16 * 4 * 4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.norm3 = nn.LayerNorm(120)
        self.fc2 = nn.Linear(120, 84)
        self.norm4 = nn.LayerNorm(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.mish(x)
        x = self.norm1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.mish(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = self.norm2(x)
        x = self.fc1(x)
        x = F.mish(x)
        x = self.norm3(x)
        x = self.fc2(x)
        x = F.mish(x)
        x = self.norm4(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, scheduler: AnySchedule, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pbar.set_postfix(
            loss=loss.item(),
            lr=optimizer.param_groups[0]["lr"],
            wd=optimizer.param_groups[0]["weight_decay"],
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["weight_decay"])


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = torch.utils.data.DataLoader(
        MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=64,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        MNIST(
            "./data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=512,
    )
    model = Net().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-3)
    scheduler = AnySchedule(optimizer, config="./config/example.toml")
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, scheduler, epoch)
        test(model, device, test_loader)


if __name__ == "__main__":
    main()
