import os
import sys
import time
from typing import Optional


class AverageMeter(object):
    """
    计算并存储平均值和当前值的计量器
    """
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    用于在训练过程中展示进度，显示当前训练的批次、损失等信息。它接受以下参数：
    参数：
        num_batches：总共的批次数量，用于格式化展示当前批次的信息。
        meters：一组用于测量和展示的 Meter 对象，可以包括损失、准确率等。
        prefix：可选的前缀字符串，用于在展示信息时添加额外的标识。
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class TextLogger(object):
    """
    将流输出写入外部文本文件。
    Args：
        filename（str）：写入流输出的文件
        stream：要从中读取的流。默认值 sys.stdout
    """
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.terminal.close()
        self.log.close()


class CompleteLogger:
    """
    记录器，具有以下功能：
        将输出写入文件并同时在控制台上显示。
        管理检查点和调试图像的目录。
    参数：
        root (str): 记录器的根目录。
        phase (str): 训练的阶段。
    """

    def __init__(self, root, args, phase='train'):
        self.root = root
        self.phase = phase
        self.checkpoint_directory = os.path.join(self.root, "checkpoints")
        self.epoch = 0

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        # redirect std out
        # now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        log_filename = os.path.join(self.root, "{}{}_a{}_lr{}_epo{}_epo{}.txt".format(args.data, str(args.targets), args.alpha, str(args.lr), str(args.epochs), str(args.second_epochs)))
        if os.path.exists(log_filename):
            os.remove(log_filename)
        self.logger = TextLogger(log_filename)
        sys.stdout = self.logger
        sys.stderr = self.logger
        if phase != 'train':
            self.set_epoch(phase)

    def set_epoch(self, epoch):
        """Set the epoch number. Please use it during training."""
        self.epoch = epoch

    def _get_phase_or_epoch(self):
        if self.phase == 'train':
            return str(self.epoch)
        else:
            return self.phase

    def get_checkpoint_path(self, name=None):
        """
        Get the full checkpoint path.
        Args:
            name (optional): the filename (without file extension) to save checkpoint.
                If None, when the phase is ``train``, checkpoint will be saved to ``{epoch}.pth``.
                Otherwise, will be saved to ``{phase}.pth``.
        """
        if name is None:
            name = self._get_phase_or_epoch()
        name = str(name)
        return os.path.join(self.checkpoint_directory, name + ".pth")

    def close(self):
        self.logger.close()
