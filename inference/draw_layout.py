import matplotlib.pyplot as plt

def draw_boxes(boxes):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for box in boxes:
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, fill=False)
        ax.add_patch(rect)

    ax.invert_yaxis()
    plt.show()
