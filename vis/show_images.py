import matplotlib.pyplot as plt
from PIL import Image

def show_imgs(X, fig_num):
    plt.figure(fig_num)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            plt.subplot2grid((4, 4), (i, j))
            plt.imshow(Image.fromarray(X[k], 'RGB'))
            k = k + 1
    # show the plot
    #pyplot.show()

