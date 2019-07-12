import numpy as np
import matplotlib.pyplot as plt

def show_image(image_array, label, fontsize=15):
    img_height = 28
    img_width = 28
    categories = ['ant', 'bear', 'bee', 'cat', 'crab', 'dragon', 'elephant', 'mouse', 'sea turtle', 'snail']
    plt.title(categories[int(label)], fontsize=fontsize)
    plt.imshow(image_array.reshape(img_height, img_width), cmap="gray")


def show_multiple_images(images_array, labels, columns=8, figsize=(15, 10)):
    fig = plt.figure(figsize=figsize)
    num = len(labels)
    rows = np.ceil(num / columns)
    for i in range(len(labels)):
        fig.add_subplot(rows, columns, i + 1)
        show_image(images_array[i], labels[i], fontsize=figsize[0])
    plt.show()
