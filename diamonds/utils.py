import os
import matplotlib.pyplot as plt

def save_plot(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path
