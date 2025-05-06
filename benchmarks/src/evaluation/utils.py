import json
import logging
import torch
import wandb
from src.evaluation.mcq import mcq_eval
from src.evaluation.retrieval import retrieval_eval
import os

def encode_text_in_batches(model, texts, tokenizer, batch_size=64, device="cuda"):
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
        all_features.append(feats)
    return torch.cat(all_features, dim=0)


def eval_imgnet(model, data, epoch, args, tb_writer=None, tokenizer=None):
    """
    Evaluate the model on ImageNet classification accuracy.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (dict): A dictionary containing data loaders for evaluation.
        epoch (int): The current epoch number.
        args (argparse.Namespace): Parsed arguments with configurations.
        tb_writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging metrics.
        tokenizer (callable, optional): Tokenizer function for text inputs.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    print("Evaluating model on ImageNet")
    metrics = {}
    device = torch.device(args.device)
    model.eval()
    
    # with open("/root/NP-CLIP/negbench/benchmarks/scripts/imagenet_classes.txt", "r") as f:
    with open("/root/NP-CLIP/negbench/benchmarks/scripts/classes.txt", "r") as f:
        class_names = [line.strip() for line in f]

    # 使用标准 ImageNet 1000 类标签
    imagenet_classes = [f"photo of a {c}" for c in class_names]  # e.g., synset mapped text labels
    with torch.no_grad():
        # 编码类标签文本
        text_features = encode_text_in_batches(model, imagenet_classes, tokenizer, batch_size=64, device=device)  # shape: [1000, d]
        # text_inputs = tokenizer(imagenet_classes).to(device)  # tokenizer outputs tokenized text
        # text_features = model.encode_text(text_inputs)        # shape: [1000, d]
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 统计
        correct1 = 0
        correct5 = 0
        total = 0
        from tqdm import tqdm
        
        # dataloader = data["imagenet-val"].dataloader
        dataloader = data["imagenet-val"]
        
        for images, labels in tqdm(dataloader, desc="Evaluating ImageNet"):
            images = images.to(device)
            labels = labels.to(device)

            # 提取图像特征
            image_features = model.encode_image(images)        # shape: [B, d]
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 相似度得分（图像特征乘以文本特征的转置）
            logits = image_features @ text_features.T          # shape: [B, 1000]

            # top-k 预测
            top1 = logits.topk(1, dim=-1).indices.squeeze()
            top5 = logits.topk(5, dim=-1).indices

            correct1 += (top1 == labels).sum().item()
            correct5 += sum([labels[i] in top5[i] for i in range(labels.size(0))])
            total += labels.size(0)

        print(f"Correct1: {correct1}, Correct5: {correct5}, Total: {total}")
        acc1 = correct1 / total * 100
        acc5 = correct5 / total * 100
        metrics['imagenet_top1'] = acc1
        metrics['imagenet_top5'] = acc5

        print(f"[ImageNet] Top-1 Acc: {acc1:.2f}%, Top-5 Acc: {acc5:.2f}%")

        if tb_writer is not None:
            tb_writer.add_scalar("eval/imagenet_top1", acc1, epoch)
            tb_writer.add_scalar("eval/imagenet_top5", acc5, epoch)

    return metrics

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    """
    Evaluate the model on multiple-choice question (MCQ) and retrieval tasks.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (dict): A dictionary containing data loaders for evaluation.
        epoch (int): The current epoch number.
        args (argparse.Namespace): Parsed arguments with configurations.
        tb_writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging metrics.
        tokenizer (callable, optional): Tokenizer function for text inputs.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    print("Evaluating model")
    metrics = {}
    device = torch.device(args.device)
    model.eval()
    
    # 分类
    # data: {'imagenet-val': DataInfo(dataloader=<torch.utils.data.dataloader.DataLoader object at 0x7f4cd9ae9d30>, sampler=None, shared_epoch=None)}
    metrics = eval_imgnet(model, data, epoch, args, tb_writer=tb_writer, tokenizer=tokenizer)
    print(f"ImageNet Top-1 Accuracy: {metrics['imagenet_top1']:.2f}%")
    print(f"ImageNet Top-5 Accuracy: {metrics['imagenet_top5']:.2f}%")


    print("Evaluating MCQ")
    mcq_metrics = mcq_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(mcq_metrics)

    print("Evaluating Retrieval")
    retrieval_metrics = retrieval_eval(model, data, args, tokenizer=tokenizer)
    metrics.update(retrieval_metrics)

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        step = None
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        log_data['epoch'] = epoch
        wandb.log(log_data)

    return metrics


def evaluate_video(model, data, epoch, args, tb_writer=None, tokenizer=None):
    """
    Evaluate the model on video-related tasks, including MCQ and video retrieval.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (dict): A dictionary containing data loaders for evaluation.
        epoch (int): The current epoch number.
        args (argparse.Namespace): Parsed arguments with configurations.
        tb_writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging metrics.
        tokenizer (callable, optional): Tokenizer function for text inputs.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    metrics = {}
    device = torch.device(args.device)
    model.eval()

    print("Evaluating MCQ")
    mcq_metrics = mcq_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(mcq_metrics)

    print("Evaluating Video Retrieval")
    retrieval_metrics = retrieval_eval(model, data, args, tokenizer=tokenizer)
    metrics.update(retrieval_metrics)

    if not metrics:
        raise ValueError("No metrics computed during evaluation.")

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        step = None
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        log_data['epoch'] = epoch
        wandb.log(log_data)

    return metrics