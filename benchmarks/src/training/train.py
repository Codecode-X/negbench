import json
import logging
import math
import os
import time
import ast

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def train_one_epoch_mixed(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    assert args.accum_freq == 1, "Other values for accum_freq are currently not supported."
    assert isinstance(loss, tuple), "loss is not a tuple"
    loss, negation_loss = loss
    freq_check = False

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()
    
    # Load dataloader_negated from data['train_negated']
    data['train'].set_epoch(epoch)
    data['train_negated'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    dataloader_negated = data['train_negated'].dataloader

    num_batches_per_epoch = (2 * dataloader.num_batches) // args.accum_freq # factor of 2 accounts for the negated samples
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in tqdm(enumerate(dataloader)):

        batch_negated = dataloader_negated.get_samples() # Note that dataloader_negated can be sampled from indefinitely

        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        # Handle both batch and batch_negated
        (images_natural, texts_natural) = batch
        (images, texts), (images_, texts_) = batch_negated
        images_negated = torch.cat((images, images_), dim=0)
        texts_negated = torch.cat((texts, texts_), dim=0)

        images_natural = images_natural.to(device=device, dtype=input_dtype, non_blocking=True)
        images_negated = images_negated.to(device=device, dtype=input_dtype, non_blocking=True)
        texts_natural = texts_natural.to(device=device, non_blocking=True)
        texts_negated = texts_negated.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        # Compute output on natural and negated example; add loss terms
        with autocast():
            model_out_natural = model(images_natural, texts_natural)
            model_out_negated = model(images_negated, texts_negated)
            logit_scale = model_out_natural["logit_scale"]
            
            if args.distill:
                with torch.no_grad():
                    dist_model_out_natural = dist_model(images_natural, texts_natural)
                    dist_model_out_negated = dist_model(images_negated, texts_negated)
                model_out_natural.update({f'dist_{k}': v for k, v in dist_model_out_natural.items()})
                model_out_negated.update({f'dist_{k}': v for k, v in dist_model_out_negated.items()})

            losses_natural = loss(**model_out_natural, output_dict=True)
            losses_negated = negation_loss(**model_out_negated, output_dict=True)

            losses = {}
            for key in losses_natural.keys():
                losses[f"nat_{key}"] = losses_natural[key]
            for key in losses_negated:
                losses[f"synth_{key}"] = losses_negated[key]

            total_loss = args.natural_weight*sum(losses_natural.values()) + args.synthetic_weight*losses_negated["total_loss"]

            losses["loss"] = total_loss
            losses["loss_natural"] = sum(losses_natural.values())
            losses["loss_synthetic"] = losses_negated["total_loss"] # TODO: verify correctness (why negative values??)

        backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Clamping logit scale (safety measure from original paper)
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        # Performance metrics update
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images_natural) + len(images_negated)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = 2 * dataloader.num_samples # Factor of 2 accounts for the negated samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # Updating and logging loss values
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = unwrap_model(model).logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )

            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )
            if num_samples == 460928:
                freq_check = True

            # TensorBoard and WandB Logging
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                # TODO: verify that 'step' is correct or even needed
                wandb.log(log_data, step=step)

            # Reset meters for each logging window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None,
                    contrastive_iter=None, mcq_iter=None):
    """
    Trains the model for one epoch.

    Depending on whether a single loss function or multiple loss functions are provided,
    it delegates the training to the appropriate helper function.

    Args:
        model (torch.nn.Module): The model to train.
        data (dict): A dictionary containing dataloaders and related data.
        loss (callable or dict): The loss function or a dictionary of loss functions.
        epoch (int): The current epoch number.
        optimizer (torch.optim.Optimizer): The optimizer for parameter updates.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        scheduler (callable): Learning rate scheduler.
        dist_model (torch.nn.Module): The model used for distillation (if applicable).
        args (argparse.Namespace): Command-line arguments and configurations.
        tb_writer (SummaryWriter, optional): TensorBoard writer for logging.
        contrastive_iter (iterator, optional): Iterator over the contrastive dataloader.
        mcq_iter (iterator, optional): Iterator over the MCQ dataloader.

    Returns:
        contrastive_iter (iterator): Updated iterator over the contrastive dataloader.
        mcq_iter (iterator): Updated iterator over the MCQ dataloader.
    """
    if isinstance(loss, dict):
        # Multiple losses and dataloaders
        return train_one_epoch_multi_loss(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer,
                                   contrastive_iter, mcq_iter)
    else:
        # Single loss and dataloader
        train_one_epoch_single_loss(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer)
        return contrastive_iter, mcq_iter

def train_one_epoch_single_loss(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    """
    Trains the model for one epoch using a single loss function and dataloader.

    Handles standard training scenarios where only one dataset and one loss function are used.

    Args:
        model (torch.nn.Module): The model to train.
        data (dict): A dictionary containing the 'train' dataloader.
        loss (callable): The loss function.
        epoch (int): The current epoch number.
        optimizer (torch.optim.Optimizer): The optimizer for parameter updates.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        scheduler (callable): Learning rate scheduler.
        dist_model (torch.nn.Module): The model used for distillation (if applicable).
        args (argparse.Namespace): Command-line arguments and configurations.
        tb_writer (SummaryWriter, optional): TensorBoard writer for logging.

    Returns:
        None
    """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # Set epoch for shuffling
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # Gradient accumulation
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            if ((i + 1) % args.accum_freq) > 0:
                continue

            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

            accum_images, accum_texts, accum_features = [], [], {}

        # Optimizer step and scaler update
        optimizer_step_and_update(optimizer, scaler, model, args)

        # Logging and metrics update
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # Update loss meters
            update_loss_meters(losses_m, losses, batch_size)

            # Logging
            log_training_progress(
                epoch, num_samples, samples_per_epoch, percent_complete,
                data_time_m, batch_time_m, optimizer, logit_scale, losses_m,
                args, step, tb_writer
            )

            batch_time_m.reset()
            data_time_m.reset()
    # End of training loop

def train_one_epoch_multi_loss(model, data, loss_dict, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None,
                               contrastive_iter=None, mcq_iter=None):
    """
    Trains the model for one epoch using multiple loss functions and dataloaders.

    Handles scenarios where multiple datasets and corresponding loss functions are used,
    such as combining contrastive loss with an MCQ loss.

    Args:
        model (torch.nn.Module): The model to train.
        data (dict): A dictionary containing dataloaders (e.g., 'train' and 'mcq_train').
        loss_dict (dict): A dictionary of loss functions with keys corresponding to tasks.
        epoch (int): The current epoch number.
        optimizer (torch.optim.Optimizer): The optimizer for parameter updates.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        scheduler (callable): Learning rate scheduler.
        dist_model (torch.nn.Module): The model used for distillation (if applicable).
        args (argparse.Namespace): Command-line arguments and configurations.
        tb_writer (SummaryWriter, optional): TensorBoard writer for logging.
        contrastive_iter (iterator, optional): Iterator over the contrastive dataloader.
        mcq_iter (iterator, optional): Iterator over the MCQ dataloader.

    Returns:
        contrastive_iter (iterator): Updated iterator over the contrastive dataloader.
        mcq_iter (iterator): Updated iterator over the MCQ dataloader.
    """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    # Retrieve loss functions
    loss_contrastive = loss_dict['contrastive']
    loss_mcq = loss_dict['mcq']

    # Set epochs and retrieve dataloaders
    data['train'].set_epoch(epoch)
    data['mcq_train'].set_epoch(epoch)
    contrastive_dataloader = data['train'].dataloader if args.contrastive_loss_weight > 0 else None
    mcq_dataloader = data['mcq_train'].dataloader if args.mcq_loss_weight > 0 else None

    # Use provided iterators or create new ones if None
    if contrastive_iter is None and contrastive_dataloader:
        contrastive_iter = iter(contrastive_dataloader)
    if mcq_iter is None and mcq_dataloader:
        mcq_iter = iter(mcq_dataloader)

    # Determine number of batches
    num_batches_list = []
    if contrastive_dataloader:
        num_batches_list.append(len(contrastive_dataloader))
    if mcq_dataloader:
        num_batches_list.append(len(mcq_dataloader))
    # TODO: decide whether to stop at min or max
    num_batches_per_epoch = min(num_batches_list) if num_batches_list else 0

    # Percentage of batches_per_epoch to keep is 50% (now changed to 35%)
    num_batches_per_epoch = int(num_batches_per_epoch * 0.35) # TODO: should this be a parameter?

    if num_batches_per_epoch == 0:
        logging.warning("No active loss functions or dataloaders are available for training.")
        return  # Exit the function as there's nothing to train

    # Calculate the frequency to update tqdm
    update_frequency = max(1, num_batches_per_epoch // 20)

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # Progress bar for tracking
    pbar = tqdm(range(num_batches_per_epoch), desc=f"Epoch {epoch}", dynamic_ncols=True)
    for i in pbar:
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        total_loss = 0.0
        losses = {}

        total_batch_size = 0  # Initialize total batch size
        
        # Process contrastive batch if applicable
        if args.contrastive_loss_weight > 0:
            contrastive_iter, contrastive_loss_value, batch_size_contrastive = process_contrastive_batch(
                model, loss_contrastive, contrastive_dataloader, contrastive_iter,
                device, input_dtype, autocast, args, dist_model
            )
            if contrastive_loss_value is not None:
                total_loss += contrastive_loss_value * args.contrastive_loss_weight
                losses['contrastive_loss'] = (contrastive_loss_value / args.contrastive_loss_weight).detach()
                total_batch_size += batch_size_contrastive
            else:
                batch_size_contrastive = 0 # Set batch size to 0 if no loss was computed

        # Process MCQ batch if applicable
        if args.mcq_loss_weight > 0:
            mcq_iter, mcq_loss_value, batch_size_mcq = process_mcq_batch(
                model, loss_mcq, mcq_dataloader, mcq_iter,
                device, input_dtype, autocast, args
            )
            if mcq_loss_value is not None:
                total_loss += mcq_loss_value * args.mcq_loss_weight
                losses['mcq_loss'] = (mcq_loss_value / args.mcq_loss_weight).detach()
                total_batch_size += batch_size_mcq
            else:
                batch_size_mcq = 0 # Set batch size to 0 if no loss was computed

        if total_loss == 0.0 or total_batch_size == 0:
            continue  # Skip this iteration if no loss was computed

        losses["loss"] = total_loss.detach()

        # Update tqdm every 1/20th of num_batches_per_epoch
        if i % update_frequency == 0:
            pbar.set_postfix({
                "contrastive_loss": losses.get("contrastive_loss", 0.0).item(),
                "mcq_loss": losses.get("mcq_loss", 0.0).item(),
                "total_loss": total_loss.item()
            })

        # Backward pass
        backward(total_loss, scaler)

        # Optimizer step and scaler update
        optimizer_step_and_update(optimizer, scaler, model, args)

        # Logging and metrics update
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            # Compute batch size
            batch_size = compute_total_batch_size(locals())
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = num_batches_per_epoch * batch_size * args.world_size
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # Update loss meters
            update_loss_meters(losses_m, losses, total_batch_size)

            # Logging
            logit_scale = unwrap_model(model).logit_scale
            log_training_progress(
                epoch, num_samples, samples_per_epoch, percent_complete,
                data_time_m, batch_time_m, optimizer, logit_scale, losses_m,
                args, step, tb_writer
            )

            batch_time_m.reset()
            data_time_m.reset()

    return contrastive_iter, mcq_iter  # Return updated iterators
    # End of training loop


def process_contrastive_batch(model, loss_contrastive, contrastive_dataloader, contrastive_iter, device, input_dtype, autocast, args, dist_model):
    """
    Processes a batch from the contrastive dataloader and computes the contrastive loss.
    Resets the iterator if the dataloader is exhausted.

    Args:
        model (torch.nn.Module): The model to train.
        loss_contrastive (callable): The contrastive loss function.
        contrastive_dataloader (DataLoader): The contrastive dataloader.
        contrastive_iter (iterator): Iterator over the contrastive dataloader.
        device (torch.device): The device to run computations on.
        input_dtype (torch.dtype): The data type for inputs.
        autocast (context manager): Context manager for mixed precision.
        args (argparse.Namespace): Command-line arguments and configurations.
        dist_model (torch.nn.Module): The model used for distillation (if applicable).

    Returns:
        contrastive_iter (iterator): Updated iterator over the contrastive dataloader.
        contrastive_loss_value (torch.Tensor or None): The computed contrastive loss value.
        batch_size (int): The size of the batch used to compute the loss.
    """
    try:
        images_contrastive, texts_contrastive = next(contrastive_iter)
    except StopIteration:
        # Reset the iterator and get the next batch
        contrastive_iter = iter(contrastive_dataloader)
        images_contrastive, texts_contrastive = next(contrastive_iter)

    images_contrastive = images_contrastive.to(device=device, dtype=input_dtype, non_blocking=True)
    texts_contrastive = texts_contrastive.to(device=device, non_blocking=True)
    with autocast():
        model_out_contrastive = model(images_contrastive, texts_contrastive)
        logit_scale = model_out_contrastive["logit_scale"]

        if args.distill:
            with torch.no_grad():
                dist_model_out_contrastive = dist_model(images_contrastive, texts_contrastive)
            model_out_contrastive.update({f'dist_{k}': v for k, v in dist_model_out_contrastive.items()})

        # Compute contrastive loss
        losses_contrastive = loss_contrastive(**model_out_contrastive, output_dict=True)
        contrastive_loss_value = sum(losses_contrastive.values())

    batch_size = images_contrastive.size(0)

    return contrastive_iter, contrastive_loss_value, batch_size


def process_mcq_batch(model, loss_mcq, mcq_dataloader, mcq_iter, device, input_dtype, autocast, args):
    """
    Processes a batch from the MCQ dataloader and computes the MCQ loss.
    Resets the iterator if the dataloader is exhausted.

    Args:
        model (torch.nn.Module): The model to train.
        loss_mcq (callable): The MCQ loss function.
        mcq_dataloader (DataLoader): The MCQ dataloader.
        mcq_iter (iterator): Iterator over the MCQ dataloader.
        device (torch.device): The device to run computations on.
        input_dtype (torch.dtype): The data type for inputs.
        autocast (context manager): Context manager for mixed precision.
        args (argparse.Namespace): Command-line arguments and configurations.

    Returns:
        mcq_iter (iterator): Updated iterator over the MCQ dataloader.
        mcq_loss_value (torch.Tensor or None): The computed MCQ loss value.
        batch_size (int): The size of the batch used to compute the loss.
    """
    try:
        images_mcq, captions_tokens, correct_answers, _ = next(mcq_iter)
    except StopIteration:
        # Reset the iterator and get the next batch
        mcq_iter = iter(mcq_dataloader)
        images_mcq, captions_tokens, correct_answers, _ = next(mcq_iter)

    images_mcq = images_mcq.to(device=device, dtype=input_dtype, non_blocking=True)
    correct_answers = correct_answers.to(device=device, dtype=torch.long, non_blocking=True)

    batch_size_mcq, num_options, seq_length = captions_tokens.size()
    captions_tokens = captions_tokens.view(batch_size_mcq * num_options, seq_length).to(device=device, non_blocking=True)

    with autocast():
        # Encode images and text
        image_features_mcq = model.encode_image(images_mcq)
        text_features_mcq = model.encode_text(captions_tokens)

        # Reshape text features
        feature_dim = text_features_mcq.size(-1)
        text_features_mcq = text_features_mcq.view(batch_size_mcq, num_options, feature_dim)

        # Use the same logit scale
        logit_scale_mcq = model.logit_scale.exp()

        # Compute MCQ loss
        mcq_loss_value = loss_mcq(image_features_mcq, text_features_mcq, correct_answers, logit_scale_mcq)

    batch_size = images_mcq.size(0)

    return mcq_iter, mcq_loss_value, batch_size


def optimizer_step_and_update(optimizer, scaler, model, args):
    """
    Performs an optimizer step and updates the gradient scaler.

    Handles gradient clipping if specified and ensures proper scaling in mixed precision training.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for parameter updates.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        model (torch.nn.Module): The model being trained.
        args (argparse.Namespace): Command-line arguments and configurations.

    Returns:
        None
    """
    if scaler is not None:
        if args.horovod:
            optimizer.synchronize()
            scaler.unscale_(optimizer)
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            with optimizer.skip_synchronize():
                scaler.step(optimizer)
        else:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
        scaler.update()
    else:
        if args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        optimizer.step()

    # Clamp logit scale
    with torch.no_grad():
        unwrap_model(model).logit_scale.clamp_(0, math.log(100))

def compute_total_batch_size(local_vars):
    """
    Computes the total batch size from the available local variables.

    Used to aggregate batch sizes when multiple dataloaders are used.

    Args:
        local_vars (dict): Dictionary of local variables from the calling function.

    Returns:
        int: The total batch size.
    """
    batch_size = 0
    if 'images_contrastive' in local_vars:
        batch_size += local_vars['images_contrastive'].size(0)
    if 'images_mcq' in local_vars:
        batch_size += local_vars['images_mcq'].size(0)
    return batch_size

def update_loss_meters(losses_m, losses, batch_size):
    """
    Updates the loss meters with the latest batch losses.

    Maintains running averages and current values for each loss component.

    Args:
        losses_m (dict): Dictionary of AverageMeter instances for each loss.
        losses (dict): Dictionary of the latest loss values.
        batch_size (int): The size of the batch used to compute the losses.

    Returns:
        None
    """
    if batch_size == 0:
        return  # Skip updating if batch_size is zero
    for key, val in losses.items():
        if key not in losses_m:
            losses_m[key] = AverageMeter()
        losses_m[key].update(val.item(), batch_size)

def log_training_progress(
    epoch, num_samples, samples_per_epoch, percent_complete,
    data_time_m, batch_time_m, optimizer, logit_scale, losses_m,
    args, step, tb_writer
):
    """
    Logs training progress, including losses, learning rates, and performance metrics.

    Outputs logs to the console and optionally to TensorBoard and WandB.

    Args:
        epoch (int): The current epoch number.
        num_samples (int): The number of samples processed so far.
        samples_per_epoch (int): Total number of samples in an epoch.
        percent_complete (float): Percentage of the epoch completed.
        data_time_m (AverageMeter): Meter for data loading time.
        batch_time_m (AverageMeter): Meter for batch processing time.
        optimizer (torch.optim.Optimizer): The optimizer used.
        logit_scale (torch.Tensor or float): The logit scaling factor.
        losses_m (dict): Dictionary of AverageMeter instances for each loss.
        args (argparse.Namespace): Command-line arguments and configurations.
        step (int): The global step count.
        tb_writer (SummaryWriter, optional): TensorBoard writer for logging.

    Returns:
        None
    """
    logit_scale_scalar = logit_scale.item() if isinstance(logit_scale, torch.Tensor) else logit_scale
    loss_log = " ".join(
        [
            f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
            for loss_name, loss_m in losses_m.items()
        ]
    )
    samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
    samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
    logging.info(
        f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] "
        f"Data (t): {data_time_m.avg:.3f} "
        f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
        f"LR: {optimizer.param_groups[0]['lr']:5f} "
        f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
    )

    # Logging data
    log_data = {
        "data_time": data_time_m.val,
        "batch_time": batch_time_m.val,
        "samples_per_second": samples_per_second,
        "samples_per_second_per_gpu": samples_per_second_per_gpu,
        "scale": logit_scale_scalar,
        "lr": optimizer.param_groups[0]["lr"]
    }
    log_data.update({name: val.val for name, val in losses_m.items()})
    log_data = {"train/" + name: val for name, val in log_data.items()}

    if tb_writer is not None:
        for name, val in log_data.items():
            tb_writer.add_scalar(name, val, step)

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        log_data['step'] = step
        wandb.log(log_data, step=step)


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
    
def create_one_hot_encoding(class_label_lists, total_classes):
    labels_flat = np.concatenate(class_label_lists)
    labels_flat = labels_flat.astype(np.int64)
    sample_indices = np.repeat(np.arange(len(class_label_lists)), [len(sublist) for sublist in class_label_lists])
    one_hot_matrix = np.zeros((len(class_label_lists), total_classes), dtype=int)
    one_hot_matrix[sample_indices, labels_flat] = 1
    
    return one_hot_matrix

def compute_label_relations(image_labels, caption_pos_labels, caption_neg_labels, num_classes):
  image_one_hot = create_one_hot_encoding(image_labels, num_classes)
  caption_pos_one_hot = create_one_hot_encoding(caption_pos_labels, num_classes)
  caption_neg_one_hot = create_one_hot_encoding(caption_neg_labels, num_classes)

  positive_relations = ((image_one_hot[None, :, :] - caption_pos_one_hot[:, None, :]) < 0).sum(axis=2) == 0
  negative_relations = ((1 - image_one_hot[None, :, :] - caption_neg_one_hot[:, None, :]) < 0).sum(axis=2) == 0
  combined_relations = positive_relations & negative_relations

  return {
        "positive_relations": positive_relations,
        "negative_relations": negative_relations,
        "combined_relations": combined_relations
    }

def get_precision(tp, fp, tn=None, fn=None):
    return np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)

def get_recall(tp, fp=None, tn=None, fn=None):
    return np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)

def get_f1(tp, fp, tn, fn):
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn=fn)
    return 2 * np.divide(precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)

def get_macro_acc(tp, fp, tn, fn):
    p_acc = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    n_acc = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
    return (p_acc + n_acc) / 2

def get_micro_acc(tp, fp, tn, fn):
    return np.divide(tp + tn, tp + fp + tn + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fp + tn + fn) != 0)

def get_binary_metrics(tp, fp, tn, fn):
    result = {} 
    # result = {
    #     "macro_acc_per_caption": get_macro_acc(tp, fp, tn, fn),
    #     "micro_acc_per_caption": get_micro_acc(tp, fp, tn, fn), 
    #     "F1_score_per_caption": get_f1(tp, fp, tn, fn),
    #     "Precision_per_caption": get_precision(tp, fp),
    #     "Recall_per_caption": get_recall(tp, fn=fn)
    # }

    # result["avg_macro_acc"] = result["macro_acc_per_caption"].mean()
    # result["avg_micro_acc"] = result["micro_acc_per_caption"].mean()
    # result["avg_F1_score"] = result["F1_score_per_caption"].mean()

    result["avg_macro_acc"] = get_macro_acc(tp, fp, tn, fn).mean()
    result["avg_micro_acc"] = get_micro_acc(tp, fp, tn, fn).mean()
    result["avg_F1_score"] = get_f1(tp, fp, tn, fn).mean()

    return result


def find_optimal_threshold(scores, labels, metric_func=get_macro_acc):
    sorted_indices = np.argsort(scores, axis=1)
    sorted_scores = np.take_along_axis(scores, sorted_indices, axis=1)
    sorted_labels = np.take_along_axis(labels, sorted_indices, axis=1)

    t_labels = sorted_labels[:, :, None]
    above_th = sorted_scores[:, :, None] >= sorted_scores[:, None, :]

    tp = np.sum(above_th & t_labels, axis=1)
    fp = np.sum(above_th & ~t_labels, axis=1)
    fn = np.sum(~above_th & t_labels, axis=1)
    tn = np.sum(~above_th & ~t_labels, axis=1)
  
    metric_values = metric_func(tp, fp, tn, fn)
    max_idx = np.argmax(metric_values, axis=1)
    row_indices = np.arange(scores.shape[0])
    
    optimal_metrics = get_binary_metrics(tp[row_indices, max_idx], fp[row_indices, max_idx], 
                                         tn[row_indices, max_idx], fn[row_indices, max_idx])
    
    return optimal_metrics

def get_test_prompt_metrics(image_features, image_positive_labels, test_prompts, prompts_positive_labels, prompts_negative_labels, results={}, num_classes=80):
    image_positive_labels = [ast.literal_eval(item) for item in image_positive_labels]
    prompts_positive_labels = [ast.literal_eval(item) for item in prompts_positive_labels]
    prompts_negative_labels = [ast.literal_eval(item) for item in prompts_negative_labels]

    labels = compute_label_relations(image_positive_labels, prompts_positive_labels,
                                      prompts_negative_labels, num_classes)
    
    if torch.is_tensor(image_features) and torch.is_tensor(test_prompts):
        image_features = image_features.cpu().numpy()
        test_prompts = test_prompts.cpu().numpy()
    elif isinstance(image_features, np.ndarray) and isinstance(test_prompts, np.ndarray):
        # They're already NumPy arrays, do nothing
        pass
    else:
        # Handle other types if necessary, or raise an error
        raise TypeError("The variable must be a PyTorch tensor or a NumPy array.")
    
    scores = test_prompts@image_features.T
    for relation in ['positive_relations', 'negative_relations', 'combined_relations']:
        relation_result = find_optimal_threshold(scores, labels[relation], metric_func=get_macro_acc)
        for key in relation_result.keys():
            results[f'{relation}_{key}'] = relation_result[key]

    return results
