import DomainBed.domainbed.datasets as dbdatasets
import DomainBed.domainbed.lib.misc as misc

"""
    外部：get_domainbed_datasets 方法
    内部：get_subdatasets 方法
"""


def get_subdatasets(dataset, class_keys):
    """
    根据给定的类别键划分子集
    Args:
        dataset: 输入的数据集对象
        class_keys: 需要划分为基础集（base_dataset）的类别键列表
    Returns:
        base_dataset: 划分后的基础数据集，包含指定类别键的样本
        open_dataset: 划分后的开放数据集，包含除了指定类别键以外的样本
    """
    base_keys, open_keys = [], []
    # 遍历数据集的样本
    for i, (_, label) in enumerate(dataset.samples):
        if label in class_keys:
            base_keys.append(i)
        else:
            open_keys.append(i)
    # 根据基础集和开放集的键列表创建对应的子集
    base_dataset = misc._SplitDataset(dataset, base_keys)
    open_dataset = misc._SplitDataset(dataset, open_keys)
    return base_dataset, open_dataset


def get_domainbed_datasets(dataset_name, root, targets, holdout=0.2, open_ratio=0):
    """
    获取 DomainBed 数据集的子集
    Args:
        dataset_name (str): 数据集名称
        root (str): 数据集根目录
        targets (list): 目标域的索引列表
        holdout (float): 验证集的比例
        open_ratio (float): 开放集的比例
    Returns:
        train_datasets (list): 训练数据集列表
        val_datasets (list): 验证数据集列表
        test_datasets (list): 测试数据集列表
        open_datasets (list): 开放数据集列表
        base_class_names (list): 基础类别的名称列表
        open_class_names (list): 开放类别的名称列表
    """

    assert dataset_name in vars(dbdatasets)  # 确保参数 dataset_name 在 dbdatasets 模块的变量中存在，否则引发 AssertionError 异常

    hparams = {"data_augmentation": True}  # 超参列表
    datasets = vars(dbdatasets)[dataset_name](root, targets, hparams)
    class_names = datasets[0].classes
    if open_ratio > 0:
        # Sample subclasses
        keys = list(range(len(class_names)))
        base_class_keys = keys[:int((1 - open_ratio) * len(keys))]
        base_class_names = [class_name for i, class_name in enumerate(class_names) if i in base_class_keys]
        open_class_names = [class_name for class_name in class_names if class_name not in base_class_names]
        in_bases, in_opens, out_bases, out_opens = [], [], [], []
        for env_i, env in enumerate(datasets):
            base_env, open_env = get_subdatasets(env, base_class_keys)
            out_base, in_base = misc.split_dataset(base_env, int(len(base_env) * holdout), misc.seed_hash(0, env_i, "base"))
            out_open, in_open = misc.split_dataset(open_env, int(len(open_env) * holdout), misc.seed_hash(0, env_i, "open"))
            in_bases.append(in_base)
            in_opens.append(in_open)
            out_bases.append(out_base)
            out_opens.append(out_open)
        train_datasets = [d for (i, d) in enumerate(in_bases) if i not in targets]
        val_datasets = [d for (i, d) in enumerate(out_bases) if i not in targets]
        test_datasets = [d for (i, d) in enumerate(in_bases) if i in targets] + [d for (i, d) in enumerate(out_bases) if i in targets]
        open_datasets = [d for (i, d) in enumerate(in_opens) if i in targets] + [d for (i, d) in enumerate(out_opens) if i in targets]
        return train_datasets, val_datasets, test_datasets, open_datasets, base_class_names, open_class_names
    else:
        in_splits, out_splits = [], []
        for env_i, env in enumerate(datasets):
            out, in_ = misc.split_dataset(env,
                int(len(env) * holdout),
                misc.seed_hash(0, env_i))
            in_splits.append(in_)
            out_splits.append(out)
        train_datasets = [d for (i, d) in enumerate(in_splits) if i not in targets]
        val_datasets = [d for (i, d) in enumerate(out_splits) if i not in targets]
        test_datasets = [d for (i, d) in enumerate(out_splits) if i in targets]
        return train_datasets, val_datasets, test_datasets, class_names
