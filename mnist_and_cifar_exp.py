import argparse
import os
import random
from copy import deepcopy
from datetime import datetime

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from GeneralizedAdadelta import BaseModel, get_adadelta, get_adadeltaS


class Network(BaseModel):
    def __init__(self):
        super(Network, self).__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv1 = torch.nn.Conv2d(1, 32, 5, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, bias=False)
        self.linear1 = torch.nn.Linear(64 * 5 * 5, 1024, bias=False)
        self.linear2 = torch.nn.Linear(1024, 10, bias=False)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Net_vgg(BaseModel):
    def __init__(self):
        super(Net_vgg, self).__init__()

        self.pool = torch.nn.MaxPool2d(2, 2)

        channels = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(channels[i], channels[i + 1], 3, padding="same", bias=(i == 0)) for i in range(len(channels) - 1)]
        )

        self.linear = torch.nn.Linear(512, 10, bias=False)

    def forward(self, x):

        for i, conv in enumerate(self.convs):
            x = torch.nn.functional.relu(conv(x))
            if i in {1, 3, 6, 9, 12}:  # Apply pooling after specific layers
                x = self.pool(x)
        return self.linear(x.view(-1, 512))


def compute_batch_margin(output, y):
    output_for_correct_label = output[range(output.size(0)), y]
    mask = torch.zeros_like(output, dtype=torch.bool).scatter_(1, y.unsqueeze(1), 1)
    max_remaining_output, _ = torch.max(output[~mask].view(output.size(0), output.size(1) - 1), 1)
    return (output_for_correct_label - max_remaining_output).min().item()


def training(models, optimizers, epochs, trainloader=None, testloader=None):
    loss_fn = torch.nn.CrossEntropyLoss()

    test_accuracy_logs = [[] for _ in range(len(models))]
    training_loss_log = []
    margin_log = []
    training_accuracy_log = []

    device = next(models[0].parameters()).device

    for _ in tqdm(range(epochs)):
        for model in models:
            model.train()
        running_loss = [0 for _ in range(len(models))]
        running_correct = [0 for _ in range(len(models))]
        margins = [[] for _ in range(len(models))]
        count = 0  # count number of batches
        total = 0  # count total number of training examples
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            total += y.size(0)
            count += 1

            for j, (model, optimizer) in enumerate(zip(models, optimizers)):
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

                running_loss[j] += loss.item()

                with torch.no_grad():
                    margins[j].append(compute_batch_margin(output, y) / model.get_norm_power().item())

                    _, predicted = torch.max(output, 1)
                    running_correct[j] += (predicted == y).sum().item()

        training_loss_log.append([loss / count for loss in running_loss])
        training_accuracy_log.append([correct / total for correct in running_correct])
        margin_log.append([min(margins[j]) for j in range(len(models))])

        for model in models:
            model.eval()
        for i, (
            model,
            test_accuracy_log,
        ) in enumerate(zip(models, test_accuracy_logs)):
            with torch.no_grad():
                # compute test accuracy
                correct = 0
                total = 0
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                test_accuracy_log.append(correct / total)

    # transpose training_loss_log, training_accuracy_log and margin_log
    training_loss_log = list(map(list, zip(*training_loss_log)))
    training_accuracy_log = list(map(list, zip(*training_accuracy_log)))
    margin_log = list(map(list, zip(*margin_log)))
    return training_accuracy_log, test_accuracy_logs, training_loss_log, margin_log


def save_results(
    training_accuracy_logs,
    test_accuracy_logs,
    training_loss_log,
    margin_logs,
    folder,
    prefix="",
):
    for data, name in [
        (margin_logs, "margins"),
        (training_accuracy_logs, "training_accuracy"),
        (test_accuracy_logs, "test_accuracy"),
        (training_loss_log, "training_loss"),
    ]:
        df = pd.DataFrame(data).T
        df.columns = ["AdaDeltaNS", "AdaDeltaN", "AdaDeltaS", "AdaDelta", "SGD"]
        df.index += 1
        df.index.names = ["epoch"]
        df.to_csv(os.path.join(folder, f"{prefix}{name}.csv"), index=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training options for the experiment.")
    parser.add_argument(
        "--experiment",
        choices=["mnist", "cifar"],
        required=True,
        help="Choose the dataset for the experiment: mnist or cifar.",
    )
    parser.add_argument(
        "--mode",
        choices=["fine", "coarse"],
        required=True,
        help="Choose the mode for the experiment: fine or coarse.",
    )
    parser.add_argument("--device", default="cuda", help="Device to use for training (default: cuda).")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs for the experiment (default: 1).",
    )

    args = parser.parse_args()

    dataset_size = 60000 if args.experiment == "mnist" else 50000
    epochs = 500 if args.experiment == "mnist" else 1000

    stability_terms_std = 1.0 if args.experiment == "mnist" else 0.25

    batch_scale = 1.0 if args.mode == "fine" else 2.5 if args.experiment == "cifar" else 10.0
    batch_size = round(100 * batch_scale)
    total_steps = round(epochs * dataset_size / 100)

    device = args.device

    prefix = f"{args.experiment}_{args.mode}_"

    sgd_lr = 0.01 * batch_scale if args.experiment == "mnist" else 0.1 * batch_scale
    adadeltas_lr = 0.1 * batch_scale

    print(f"Running {args.runs} experiments with {args.experiment} dataset in {args.mode} mode on {device} device.")
    print(
        f"Batch size: {batch_size}, total steps: {total_steps}, SGD learning rate: {sgd_lr}, AdaDelta learning rate: {adadeltas_lr}, Epochs: {epochs}, Stability terms std: {stability_terms_std}"
    )

    os.makedirs("runs", exist_ok=True)

    results = []
    for i in range(args.runs):
        model = Network() if args.experiment == "mnist" else Net_vgg()
        models = [deepcopy(model).to(device) for _ in range(5)]

        adadeltaNS = get_adadeltaS(models[0].parameters(), scale=stability_terms_std, total_steps=total_steps, lr=adadeltas_lr)
        adadeltaNS_sd = adadeltaNS.state_dict()
        adadeltaN_sd = deepcopy(adadeltaNS_sd)
        for group in adadeltaN_sd["param_groups"]:
            group["rho"] = iter([0.9 for _ in range(total_steps)])
        adadeltaN = get_adadelta(models[1].parameters(), scale=stability_terms_std, lr=adadeltas_lr)
        adadeltaN.load_state_dict(adadeltaN_sd)

        adadeltaS = get_adadeltaS(models[2].parameters(), scale=0.0, total_steps=total_steps, lr=adadeltas_lr)
        adadelta = get_adadelta(models[3].parameters(), scale=0.0, lr=adadeltas_lr)
        sgd = torch.optim.SGD(models[4].parameters(), lr=sgd_lr)
        optimizers = [adadeltaNS, adadeltaN, adadeltaS, adadelta, sgd]

        if args.experiment == "mnist":
            transform = transforms.Compose([transforms.ToTensor()])
            trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        training_results = training(
            models,
            optimizers,
            epochs=epochs,
            trainloader=trainloader,
            testloader=testloader,
        )

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        random_number = random.randint(10000, 99999)
        folder = f"runs/{current_time}_{random_number}_{prefix}{i}"
        os.makedirs(folder)

        save_results(*training_results, folder=folder)
