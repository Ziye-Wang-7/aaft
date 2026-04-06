import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset

from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers import OFATokenizer, OFAModel

import converter_domainbed
import converter_dassl

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


#----------------------------
# COOp
#----------------------------
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
class PromptLearner(nn.Module):
    def __init__(self, template, classnames, clip_model, position, device):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = template
        dtype = clip_model.dtype

        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        prompt_prefix = ctx_init

        print(f'Initial context: "{prompt_prefix}" Position:"{position}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts


class ContentCLIP(nn.Module):
    def __init__(self, classnames, template, clip_model, image_encoder, device, position):
        super().__init__()
        self.prompt_learner = PromptLearner(template, classnames, clip_model, position, device).to(device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = image_encoder
        self.text_encoder = TextEncoder(clip_model).to(device)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device

    def forward(self, x):
        #
        image_features = self.image_encoder(x.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        with torch.no_grad():
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features


#----------------------------
# 字幕生成
#----------------------------
class ImageCaption(nn.Module):
    def __init__(self, device):
        super(ImageCaption, self).__init__()
        self.device = device
        path_to_ofa = './OFA-base/'
        self.tokenizer = OFATokenizer.from_pretrained(path_to_ofa)
        self.model = OFAModel.from_pretrained(path_to_ofa, use_cache=False).to(device)

    def forward(self, x, labels):
        resized_xs = TF.resize(x, (384, 384))
        texts = []
        for resized_x, label in zip(resized_xs, labels):
            txt = 'what dose this image of {} describe?'.format(label)
            with torch.no_grad():
                inputs = self.tokenizer([txt], return_tensors="pt").input_ids.to(self.device)
                gen = self.model.generate(inputs, patch_images=resized_x.unsqueeze(0), num_beams=5, no_repeat_ngram_size=3)
                text = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            texts.append(''.join(text))
        return texts

#----------------------------
# 权重对齐更新
#----------------------------
class WeightAlignModel(nn.Module):
    def __init__(self, model, score, device):
        super(WeightAlignModel, self).__init__()
        self.merged_model = copy.deepcopy(model).to(device)
        self.last_score = score  # 用于存储前一 epoch 的对齐分数
        self.device = device

    def forward(self, model, score):
        print(' * Updata parameters by last_score:', self.last_score, 'score:', score)
        if self.last_score == 0:  # 预训练模型与第一步模型
            for merged_param, param in zip(self.merged_model.parameters(), model.parameters()):
                merged_param.data = merged_param * 0.5 + param * 0.5
        else:
            for merged_param, param in zip(self.merged_model.parameters(), model.parameters()):
                merged_param.data = (merged_param * self.last_score + param * score) / (self.last_score + score)
        self.last_score += score



#----------------------------
# 计算自蒸馏损失
#----------------------------
class SelfKDLossModel(nn.Module):
    def __init__(self, temperature, device):
        super(SelfKDLossModel, self).__init__()
        self.previous_probs = None  # 用于存储前一 epoch 的概率分布
        self.previous_f = None
        self.device = device
        self.temperature = temperature

    def update_previous_probs(self, current_probs):
        # 更新前一 epoch 的概率分布
        self.previous_probs = current_probs.clone().detach().to(self.device)

    def update_previous_x(self, current_x):
        # 更新前一 epoch 的图像编码
        self.previous_f = current_x.clone().detach().to(self.device)

    def forward(self, current_f, current_probs, batch):
        if self.previous_probs is None:  # 第一个 epoch 返回 0
            return 0
        # 图图对齐损失
        ii = current_f @ self.previous_f[batch].T
        labels = torch.arange(int(ii.shape[0])).to(self.device)
        ii_loss = F.cross_entropy(ii, labels)
        # 图文对齐损失
        it = current_probs @ self.previous_probs[batch].T
        it_loss = F.cross_entropy(it, labels)
        return ii_loss + it_loss


#----------------------------
# 加载数据集
#----------------------------
def get_dataset(args):
    if args.task == "domain_shift":
        # 加载 domainbed
        train_datasets, val_datasets, test_datasets, class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets, holdout=0.2)
        train_class_names = class_names
        train_iter = DataLoader(ConcatDataset(train_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": class_names
            }
        ]
        template = "a photo of a {}."
        return train_iter, val_loader, test_loaders, train_class_names, template
    elif args.task == "open_class":
        # 加载 dassl
        train_dataset, val_dataset, test_dataset, open_dataset, base_class_names, open_class_names, template = \
            converter_dassl.get_dassl_datasets(dataset_name=args.data, root=args.root, n_shot=args.n_shot)
        train_class_names = base_class_names
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
        train_iter = ForeverDataIterator(train_loader)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(open_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": open_class_names
            }
        ]
        return train_iter, val_loader, test_loaders, train_class_names, template
    elif args.task == "in_the_wild":
        # 加载 domainbed
        train_datasets, val_datasets, test_datasets, open_datasets, base_class_names, open_class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets, holdout=0.2, open_ratio=0.5)
        train_class_names = base_class_names
        train_iter = DataLoader(ConcatDataset(train_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(ConcatDataset(open_datasets), batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            }
        ]
        template = "a photo of a {}."
        return train_iter, val_loader, test_loaders, train_class_names, template
    else:
        print("Datasets Error!!")
        
#----------------------------
# 无限数据迭代器
#----------------------------    
class ForeverDataIterator:

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)
        
       
