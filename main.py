import torch

import numpy as np

from x.pytorch import Net, get_train_dataloaders, get_test_dataloaders, transform, train, test, TrainingDataset
from x.searchspace import sample, to_json, from_json, EDF

def main() -> None:
    space: list[int] = []
    for m in sample(43, m=10):
        net = Net(m)
        space.append(net.flops)

    space: np.ndarray = np.asarray(space)
    space = np.sort(space)

    k = [np.argmax(space >= i) / len(space) for i in range(space.argmin(), space.argmax() + 1)]
    print(k)

    return
    train_ds = TrainingDataset(
        "D:\\Henrik\\Uni\\masterarbeit\\GTSRB\\GTSRB-training\\gt-training.csv",
        "D:\\Henrik\\Uni\\masterarbeit\\GTSRB\\GTSRB-training\\images",
        transform=transform(80)
    )

    train_ds.show(5)

    return
    with open("model.json", "r") as f:
        model = from_json(f.read())

    net = Net(model)

    train_dl, valid_dl = get_train_dataloaders(
        "D:\\Henrik\\Uni\\masterarbeit\\GTSRB\\GTSRB-training\\gt-training.csv",
        "D:\\Henrik\\Uni\\masterarbeit\\GTSRB\\GTSRB-training\\images",
        transform=transform(net.resolution)
    )

    train(net, 1, train_dl, valid_dl)

    test_dl = get_test_dataloaders(
        "D:\\Henrik\\Uni\\masterarbeit\\GTSRB\\GTSRB-test\\gt-test.csv",
        "D:\\Henrik\\Uni\\masterarbeit\\GTSRB\\GTSRB-test\\test",
        transform=transform(net.resolution)
    )

    accuracy = test(net, test_dl)
    print(f"Accuracy of the model is {accuracy:.2f}%")

    return
    for m in sample(43, m=1):
        with open("model.json", "w") as f:
            f.write(to_json(m))

        net = Net(m)
        print(net.flops)

        # tensor = torch.randn(1, 3, m.resolution, m.resolution)
        # output: torch.Tensor = net(tensor)

        # print(output.size())


if __name__ == "__main__":
    main()
