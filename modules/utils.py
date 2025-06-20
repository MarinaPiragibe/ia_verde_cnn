from matplotlib import pyplot as plt
import globals

def plot_sample(train_loader, data_type):
    plt.figure(figsize=(10, 5))
    dataiter = iter(train_loader)
    images, labels = next(dataiter) # Get one batch of training images

    for i in range(min(len(images), 9)): # Plot up to 9 images
        ax = plt.subplot(1, 9, i + 1)
        # PyTorch images are (C, H, W), Matplotlib expects (H, W, C)
        plt.imshow(images[i].numpy().transpose((1, 2, 0)))
        plt.title(globals.CLASS_NAMES[labels[i]], fontsize=10)
        plt.axis("off")
    plt.suptitle(f"CIFAKE {data_type} Images", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 1.5]) # Adjust layout to prevent suptitle overlap
    plt.show()