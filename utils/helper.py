import os


def get_size_dataset(dir_path):
    count = 0

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count


def draw_curve(dir_save_fig, current_epoch, x_epoch, y_loss, y_err, fig, ax0, ax1):
    x_epoch.append(current_epoch + 1)
    ax0.plot(x_epoch, y_loss['train'], 'b-', linewidth=1.0, label='train')
    ax0.plot(x_epoch, y_loss['val'], '-r', linewidth=1.0, label='val')
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax1.plot(x_epoch, y_err['train'], '-b', linewidth=1.0, label='train')
    ax1.plot(x_epoch, y_err['val'], '-r', linewidth=1.0, label='val')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("error")
    if current_epoch == 0:
        ax0.legend(loc="upper right")
        ax1.legend(loc="upper right")
    fig.savefig(os.path.join(dir_save_fig, 'train_curves.jpg'), dpi=600)
