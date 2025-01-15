import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES,\
    COCO_CLASSNAMES, VOC2007_CLASSNAMES, SIMPLEST_TEMPLATE
from .precision import get_autocast


def accuracy(output, target, topk=(1,), multiclass=False):
    if multiclass:
        acc_top1 = 0.0
        acc_top5 = 0.0

        target = [eval(x) for x in target]
        for logit, label in zip(output, target):
            true_labels = set(label)
            k, n = len(true_labels), len(logit)
            
            if k == 0:
                continue
            
            # Top-1 accuracy
            pred_labels_top1 = set(torch.topk(logit, k=k)[1].tolist())
            true_pred_labels_top1 = len(true_labels.intersection(pred_labels_top1))
            acc_top1 += true_pred_labels_top1 / k
            
            # Top-5 accuracy
            pred_labels_top5 = set(torch.topk(logit, k=min(5*k, n))[1].tolist())
            true_pred_labels_top5 = len(true_labels.intersection(pred_labels_top5))
            acc_top5 += true_pred_labels_top5 / k
        
        return acc_top1, acc_top5
    else:
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args, multiclass=False):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            if not multiclass: # Then this is a PyTorch tensor of integers (otherwise it's a list of strings)
                target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier
            acc1, acc5 = accuracy(logits, target, topk=(1, 5), multiclass=multiclass)
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    if 'synthetic-zeroshot' in data:
        top1, top5 = run(model, classifier, data['synthetic-zeroshot'].dataloader, args)
        results['synthetic-zeroshot-val-top1'] = top1
        results['synthetic-zeroshot-val-top5'] = top5

        logging.info('Finished zero-shot on synthetic dataset.')

    if 'coco-zeroshot' in data:
        logging.info('Building zero-shot classifier for COCO')
        with autocast():
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=COCO_CLASSNAMES,
                templates=SIMPLEST_TEMPLATE,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        top1, top5 = run(model, classifier, data['coco-zeroshot'].dataloader, args, multiclass=True)
        results['coco-zeroshot-top1'] = top1
        results['coco-zeroshot-top5'] = top5

        logging.info('Finished zero-shot COCO.')

    if 'voc2007-zeroshot' in data:
        logging.info('Building zero-shot classifier for VOC2007')
        with autocast():
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=VOC2007_CLASSNAMES,
                templates=SIMPLEST_TEMPLATE,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        top1, top5 = run(model, classifier, data['voc2007-zeroshot'].dataloader, args, multiclass=True)
        results['voc2007-zeroshot-top1'] = top1
        results['voc2007-zeroshot-top5'] = top5

        logging.info('Finished zero-shot VOC2007.')

    return results