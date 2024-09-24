import matplotlib.pyplot as plt
import datetime

def plot_loss_lr_and_val(train_loss, learning_rate, val_losses):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots()

        # 绘制训练损失
        ax1.plot(x, train_loss, 'r-', label='Training Loss')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.tick_params(axis='y')

        # 创建第二y轴用于绘制验证损失
        ax3 = ax1.twinx()
        ax3.plot(x, val_losses, 'b--', label='Validation Loss')
        ax3.set_ylabel('Validation Loss', color='b')
        ax3.tick_params(axis='y', labelcolor='b')

        # 创建第三y轴用于绘制学习率
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('outward', 60))
        ax2.plot(x, learning_rate, 'g-.', label='Learning Rate')
        ax2.set_ylabel('Learning Rate', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # 添加图例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper right')

        fig.tight_layout()  # 调整整体空白
        plt.show()
        # 保存图表
        fig.savefig('./loss_lr_val_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("Successfully saved the loss and learning rate curve!")
    except Exception as e:
        print(e)

def plot_map(mAP):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('Evaluation mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        plt.savefig('./mAP{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("Successfully saved mAP curve!")
    except Exception as e:
        print(e)
