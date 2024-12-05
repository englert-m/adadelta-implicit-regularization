import math
from copy import deepcopy

import pandas as pd
import torch
from tqdm import tqdm

from GeneralizedAdadelta import BaseModel, get_adadelta, get_adadeltaS


class ExponentialLoss(torch.nn.Module):
    def __init__(self):
        super(ExponentialLoss, self).__init__()

    def forward(self, output, target):
        return torch.mean(torch.exp(-output * target))


class Perceptron(BaseModel):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.5)
        self.output = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.uniform_(self.linear.weight, -1, 1)
        torch.nn.init.uniform_(self.output.weight, -1, 1)

    def forward(self, x):
        return self.output(self.leaky_relu(self.linear(x)))


def training(models, optimizers, x, y):
    loss_fn = ExponentialLoss()
    num_epochs = 5000

    accuracy_logs = [[] for _ in range(len(models))]
    loss_logs = [[] for _ in range(len(models))]

    for _ in tqdm(range(num_epochs)):
        for model, optimizer, accuracy_log, loss_log in zip(models, optimizers, accuracy_logs, loss_logs):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                accuracy = (torch.sign(output) == y).sum().item() / y.size(0)
                accuracy_log.append(accuracy)
                loss_log.append(loss.item())

    with torch.no_grad():
        normalized_margins = [(torch.min(model(x) * y) / model.get_norm_power()).item() for model in models]

        # for the first four optimizer, get the reciprocal
        reciprocals = [optimizer.get_reciprocal() for optimizer in optimizers[:4]]
        accuracy_logs = torch.stack([torch.tensor(values) for values in accuracy_logs])
        loss_logs = torch.stack([torch.tensor(values) for values in loss_logs])
    return normalized_margins, reciprocals, accuracy_logs, loss_logs


def generate_data(num_points=50):
    center = torch.tensor([math.cos(0.5), math.sin(0.5)])
    x = torch.cat(
        [
            center + torch.rand(num_points, 2) * 1.2 - 0.6,
            -center + torch.rand(num_points, 2) * 1.2 - 0.6,
        ],
        dim=0,
    )
    y = torch.cat([torch.ones(num_points, 1), -torch.ones(num_points, 1)], dim=0)
    return x, y


def save_data_to_csv(data, filename, algorithms, index_name="epoch"):
    df = pd.DataFrame(data, columns=algorithms)
    df.index += 1
    df.index.names = [index_name]
    df.to_csv(filename, index=True)


if __name__ == "__main__":
    x, y = generate_data()

    # save data
    df = pd.DataFrame(x.numpy(), columns=["x", "y"])
    df["label"] = y.numpy()
    df.to_csv("data.csv", index=False)

    results = {"margins": [], "reciprocals": [], "accuracy_logs": [], "loss_logs": []}

    for _ in range(100):
        model = Perceptron()
        models = [deepcopy(model) for _ in range(5)]

        adadeltaNS = get_adadeltaS(models[0].parameters(), scale=1.0)
        adadeltaNS_sd = adadeltaNS.state_dict()
        adadeltaN_sd = deepcopy(adadeltaNS_sd)
        for group in adadeltaN_sd["param_groups"]:
            group["rho"] = iter([0.9 for _ in range(5000)])
        adadeltaN = get_adadelta(models[1].parameters(), scale=1.0)
        adadeltaN.load_state_dict(adadeltaN_sd)

        adadeltaS = get_adadeltaS(models[2].parameters(), scale=0.0)
        adadelta = get_adadelta(models[3].parameters(), scale=0.0)
        sgd = torch.optim.SGD(models[4].parameters(), lr=0.1)
        optimizers = [adadeltaNS, adadeltaN, adadeltaS, adadelta, sgd]

        normalized_margins, reciprocals, accuracy_logs, loss_logs = training(models, optimizers, x, y)
        results["margins"].append(normalized_margins)
        results["reciprocals"].append(reciprocals)
        results["accuracy_logs"].append(accuracy_logs)
        results["loss_logs"].append(loss_logs)

    avg_accuracy = torch.mean(torch.stack(results["accuracy_logs"]), dim=0)
    avg_loss = torch.mean(torch.stack(results["loss_logs"]), dim=0)

    algorithms = ["AdaDeltaNS", "AdaDeltaN", "AdaDeltaS", "AdaDelta", "SGD"]
    save_data_to_csv(avg_accuracy.T.numpy(), "avg_accuracy.csv", algorithms)
    save_data_to_csv(avg_loss.T.numpy(), "avg_loss.csv", algorithms)
    save_data_to_csv(results["margins"], "margins.csv", algorithms, index_name="run")

    for algorithm_index, opt in enumerate(algorithms[:4]):
        df = pd.DataFrame(
            [reciprocal[algorithm_index].numpy() for reciprocal in results["reciprocals"]],
            columns=[f"weight_{i+1}" for i in range(results["reciprocals"][0][0].size(0))],
        )
        # normalize each row such that the sum of the squares is 1
        df = df.div(df.pow(2).sum(axis=1).pow(0.5), axis=0)
        df.index += 1
        df.index.names = ["run"]
        df.to_csv(f"reciprocal_{opt}.csv")
