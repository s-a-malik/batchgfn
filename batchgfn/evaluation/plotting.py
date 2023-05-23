import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch

rc = {
    "figure.constrained_layout.use": True,
    "axes.titlesize": 20,
}
sns.set_theme(style="darkgrid", palette="colorblind", rc=rc)


def plot_acquisition(
    x,
    f_samples,
    lower,
    upper,
    x_train,
    y_train,
    x_acquired,
    y_acquired,
    loss,
    mi,
    step,
    title,
    output_file=None,
):
    plt.clf()
    plt.close("all")
    fig = plt.figure(figsize=(6 * 16 / 9, 6))
    _ = plt.fill_between(
        x=x,
        y1=lower,
        y2=upper,
        alpha=0.2,
        color="C0",
    )
    for f in f_samples:
        _ = plt.plot(x, f.numpy(), color="C0", alpha=0.2)
    _ = sns.scatterplot(x=x_train, y=y_train, alpha=1.0)
    _ = sns.scatterplot(x=x_acquired, y=y_acquired, s=200)
    _ = plt.ylim(-1.5, 1.5)
    _ = plt.title(f"{title}, step {step:03d}, mse {loss:.3f}, mi {mi:.3f}")
    # plt.show()
    if output_file:
        _ = plt.savefig(output_file)
    # _ = plt.close()

    return fig


def plot_acquisition_2d_cls(
    model,
    x_pool,
    x_train,
    y_train,
    x_acquired,
    y_acquired,
    loss,
    acc,
    mi,
    step,
    title,
    output_file=None,
    device="cpu",
):
    """
    Plot the acquired points for a 2D classification problem
    Plot the decision boundaries
    TODO add labels for each class (should be obvious anyway?)
    """
    plt.clf()
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # plot decision boundary
    test_d1 = np.linspace(-10, 10, 100)
    test_d2 = np.linspace(-10, 10, 100)
    test_x_mat, test_y_mat = np.meshgrid(test_d1, test_d2)
    test_x_mat, test_y_mat = torch.Tensor(test_x_mat), torch.Tensor(test_y_mat)
    test_x = torch.cat((test_x_mat.view(-1, 1), test_y_mat.view(-1, 1)), dim=1)
    # sample means
    test_dist = model.predict(test_x.to(device))
    test_means = test_dist.mean.cpu()

    # decision boundary (argmax)
    ax.contourf(
        test_x_mat.numpy(),
        test_y_mat.numpy(),
        test_means.max(-1)[1].reshape((100, 100)),
    )
    ax.set_title(f"{title} step {step}, CE {loss:.3f}, acc {acc:.3f}, mi {mi:.3f}")
    # ax.scatter(x_pool[:, 0], x_pool[:, 1], marker="o", label="Unlabelled")
    ax.scatter(x_train[:, 0], x_train[:, 1], marker="o", c="g", s=40, label="Labelled")
    ax.scatter(
        x_acquired[:, 0], x_acquired[:, 1], marker="x", c="magenta", s=40, label="Query"
    )
    plt.legend()

    # plt.show()
    if output_file:
        _ = plt.savefig(output_file)
    # _ = plt.close()

    return fig
