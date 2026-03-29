import matplotlib.pyplot as plt


def show_feature(feature, title="Feature Map"):
    f = feature[0, 0].detach().cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.imshow(f, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()
