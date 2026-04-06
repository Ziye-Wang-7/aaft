import argparse
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

import clip

from engine import (get_dataset,
                    SelfKDLossModel,
                    ContentCLIP)
from utils import (CompleteLogger,
                   AverageMeter,
                   ProgressMeter)


# 获得归一化后的 CLIP 文本特征
def get_text_features(clip_model, template, class_names, device):
    with torch.no_grad():
        if template is None:
            texts = torch.cat([clip.tokenize(c.replace("_", " ")).to(device) for c in class_names]).to(device)  # [类别数, 77]
        else:
            texts = torch.cat([clip.tokenize(template.format(c.replace("_", " "))) for c in class_names]).to(device)  # [类别数, 77]
        text_features = clip_model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # [类别数, 512]
    return text_features


# 将模型中的所有参数的数据类型转换为 32 位浮点数
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


# 计算指定 k 值的前 k 个预测结果的准确率
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


# 评估
def validate(val_loader, model, text_features, args, device, shift=0) -> float:
    batch_time = AverageMeter('Time', ':6.3f')  # 每一个 batch 用时
    top1 = AverageMeter('Acc', ':6.2f')  # 准确率
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1],
        prefix='Test: ')
    # 评估模式
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device) - shift
            # 计算输出并归一化
            image_features = model(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # 计算相似度得分并计算损失
            output_similarity = image_features @ text_features.T
            acc1, = accuracy(output_similarity, target, topk=(1,))
            top1.update(acc1.item(), images.size(0))
            # 测量经过的时间
            batch_time.update(time.time() - end)
            end = time.time()
            # 每 print_freq 次打印到控制台
            if i % args.print_freq == 0:
                progress.display(i)
        print(' * Acc: {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


# 评估验证集和测试集
def evaluate_all(model, val_loader, train_text_features, test_loaders, args, device):
    # 在验证集上评估
    print("Evaluate on validation set...")
    val_acc1 = validate(val_loader, model, train_text_features, args, device)
    # 在测试集上评估
    test_accs = []
    for test_loader in test_loaders:
        split_name = test_loader["name"]
        print(f"Evaluate on {split_name} set...")
        test_acc = validate(test_loader["loader"], model, test_loader["text_features"], args, device)
        test_accs.append(test_acc)
    # 返回验证集上的 acc
    return val_acc1, test_accs


def main(args: argparse.Namespace):
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 是否根据输入数据的大小自动选择最适合当前硬件的最优配置
    cudnn.benchmark = True
    # 初始化记录器
    logger = CompleteLogger(args.log, args, args.phase)
    print(args)

    # 训练图片迭代器，验证图片加载器，测试图片加载器，训练类名，模板
    train_iter, val_loader, test_loaders, train_class_names, template = get_dataset(args)  # [类别数]  str

    # 加载 clip 模型
    clip_model, _ = clip.load(args.arch, device)
    # 创建 image 模型
    print("=> Using pre-trained model '{}'".format(args.arch))
    image_model = clip_model.visual  # 得到 clip 的视觉编码器
    image_model = image_model.to(device)
    clip.model.convert_weights(image_model)
    image_model.eval()  # 评估模式

    # 提取原始文的本特征
    train_text_features = get_text_features(clip_model, template, train_class_names, device)  # [类别数, 512]
    for test_loader in test_loaders:
        test_loader["text_features"] = get_text_features(clip_model, template, test_loader["class_names"], device)

    # 创建模型
    model = ContentCLIP(train_class_names, template, clip_model, image_model, device, args.position)

    # 若阶段为 train 则开始训练
    if args.phase == "train":
        # 记录所有 acc 和 loss
        val_accs, test_accs, train_losses = [], [], []
        # 定义优化器和学习率调度函数
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)  # AdamW优化器
        lr_scheduler = CosineAnnealingLR(optimizer, args.epochs)  # 余弦退火学习率调度器
        # 定义温度
        if args.temperature is None:
            args.temperature = clip_model.logit_scale.exp().item()  # 设置为 clip_model.logit_scale 的指数函数值（ logit_scale 用于缩放模型的 logits
        print('temperature:', args.temperature)
        # 加载自蒸馏损失模型
        self_kd_loss = SelfKDLossModel(args.temperature, device)
        # 评估 Zore-shot 性能，并设它为当前 best_val_acc1
        best_val_acc1, _ = evaluate_all(model.image_encoder, val_loader, train_text_features, test_loaders, args, device)
        # 开始训练
        # all_text_features = []
        for epoch in range(args.epochs):
            print("Learning rate: {:.4e}".format(lr_scheduler.get_last_lr()[0]))
            # 训练过程中的一些统计信息
            batch_time = AverageMeter('Time', ':4.2f')
            data_time = AverageMeter('Data', ':3.1f')
            losses = AverageMeter('Loss', ':3.2f')
            cls_accs = AverageMeter('Cls Acc', ':3.1f')
            progress = ProgressMeter(  # 进度条
                len(train_iter),
                [batch_time, data_time, losses, cls_accs],
                prefix="Epoch: [{}]".format(epoch))
            model.eval()  # 评估模式
            end = time.time()
            current_probs_all, f_all_skd = None, None
            # 训练一个 epoch
            for i, (x, labels) in enumerate(train_iter):  # domain
            # for i in range(args.second_epochs):  # open
                # x, labels = next(train_iter)  # open
                x, labels = x.to(device), labels.to(device)  # [batch_size, 3, 224, 224]  [batch_size]
                # 更新时间
                data_time_step = time.time() - end
                data_time.update(data_time_step)
                # ----------------------------
                # 模型
                # ----------------------------
                f, t = model(x)
                f = f / f.norm(dim=-1, keepdim=True)  # 图像编码 [batch_size, 512]
                f -= 0.3 * train_text_features[labels]
                y = f @ train_text_features.T
                y = args.temperature * y  # 概率分布 [batch_size, 类别数]
                y_cp = f @ t.T
                y_cp = args.temperature * y_cp  # 概率分布 [batch_size, 类别数]
                current_probs = F.softmax(y, dim=-1)
                #----------------------------
                # LOSS
                #----------------------------
                loss = F.cross_entropy(y, labels)  # 基础损失
                cp_loss = F.cross_entropy(y_cp, labels)  #字幕损失
                kd_loss = self_kd_loss(f, current_probs, i)  # 对齐损失
                loss = args.alpha * kd_loss + (1 - args.alpha) * loss + cp_loss  
                # ----------------------------
                # ELSE
                # ----------------------------
                # 更新 current_probs_all 和 x_all
                if i == 0:
                    current_probs_all = current_probs.unsqueeze(0).clone().detach().to(device)
                    f_all_skd = f.unsqueeze(0).clone().detach().to(device)
                else:
                    current_probs_all = torch.cat((current_probs_all, current_probs.unsqueeze(0)), dim=0)
                    f_all_skd = torch.cat((f_all_skd, f.unsqueeze(0)), dim=0)
                # 更新 loss 和 cls_accs
                losses.update(loss.item(), x.size(0))
                cls_acc = accuracy(y, labels)[0]  # 前 K 个类的 acc
                cls_accs.update(cls_acc.item(), x.size(0))
                # 计算梯度并反向更新
                optimizer.zero_grad()
                loss.backward()
                convert_models_to_fp32(model.image_encoder)
                optimizer.step()
                clip.model.convert_weights(model.image_encoder)
                lr_scheduler.step()
                # 测量经过的时间
                batch_time_step = time.time() - end
                batch_time.update(batch_time_step)
                end = time.time()
                # 每 print_freq 次打印到控制台
                if i % args.print_freq == 0:
                    progress.display(i)
                    print('kd_loss{},cp_loss{},loss{}'.format(kd_loss,cp_loss,loss))
            # 记录并输出当前 epoch 的 loss
            train_losses.append(losses.avg)
            print(" * Training epoch [{epoch}] Loss: {losses.avg:3.2f}".format(epoch=epoch, losses=losses))
            # 评估当前 epoch 模型，val_acc1 为当前验证集 acc
            val_acc1, test_acc = evaluate_all(model.image_encoder, val_loader, train_text_features, test_loaders, args, device)
            val_accs.append(val_acc1)
            test_accs += test_acc
            # 存储当前 epoch 的 图像编码 和 概率分布
            self_kd_loss.update_previous_probs(current_probs_all)
            self_kd_loss.update_previous_x(f_all_skd)
            # 保存 checkpoint
            torch.save(model.image_encoder.state_dict(), logger.get_checkpoint_path("epo{}".format(epoch)))  # 记录当前模型参数
            if val_acc1 >= best_val_acc1:  # 若当前模型 acc 更高，则更新 best_val_acc1
                shutil.copy(logger.get_checkpoint_path("epo{}".format(epoch)), logger.get_checkpoint_path('best'))  # 复制当前模型参数到 best
                best_val_acc1 = val_acc1
        # 画图
        print(train_losses, val_accs, test_accs)
        # draw_pic(args, train_losses, val_accs, test_accs)
        print("=> Training completed.")

    # 加载训练过程中最好的模型并评估
    model.image_encoder.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    print("=> Evaluate best model:")
    evaluate_all(model.image_encoder, val_loader, train_text_features, test_loaders, args, device)

    # 关闭 logger
    logger.close()


if __name__ == '__main__':
    # 命令行参数解析器
    parser = argparse.ArgumentParser(description='Baseline for Domain Generalization')
    # 数据集 参数
    parser.add_argument('root', help='root path of dataset')
    parser.add_argument('--data', default='DomainNet')
    parser.add_argument('--task', default='domain_shift', choices=['domain_shift', 'open_class', 'in_the_wild'])
    parser.add_argument('--targets', nargs='+', type=int, default=None, help='target domain(s) (DomainBed datasets only)')
    parser.add_argument('--n-shot', type=int, default=0)
    # 模型 参数
    parser.add_argument('--arch', metavar='ARCH', default='ViT-B/16')
    # 训练 参数
    parser.add_argument('--batch-size', default=4, type=int, help='mini-batch size (default: 12)')
    parser.add_argument('--weight-decay', dest='wd', default=0.1, type=float, help='weight decay (default: 0.1)')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--log', type=str, default='a_terra', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help="When phase is 'test', only test the model.")
    parser.add_argument('--temperature', type=float, default=None, help="Use CLIP's original temperature in default.")
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--lr', '--learning-rate', dest='lr', default=5e-6, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=12, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--position', default='end', type=str)
    parser.add_argument('--second_epochs', default=500, type=int)
    # 解析参数
    args = parser.parse_args()
    main(args)
