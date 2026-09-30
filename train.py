import logging
import os
from pathlib import Path

import torch

from cache_utils import validate_cache
from compare import assert_reference_config
from rde_runtime import (activate_rde, file_digest, restore_rng, seed_everything,
                         state_digest, training_signature, write_json)
from sampler import row_fingerprint

activate_rde()
from datasets import build_dataloader
from model import build_model
from processor.processor import do_train
from solver import build_lr_scheduler, build_optimizer
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.options import get_args


def main():
    args = get_args()
    assert_reference_config(vars(args))
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA; run the Kaggle notebook')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run one independent process per GPU, not DDP')
    args.distributed = False
    args.training = True
    if args.noisy_rate != 0:
        raise ValueError('The old reference uses noisy_rate=0.0')
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('RDE', save_dir=str(output), if_train=True, distributed_rank=0)
    logger.info(str(args).replace(',', '\n'))
    train_loader, val_img, val_txt, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    signature = training_signature(args)
    signature.update(train_rows=row_fingerprint(train_loader.dataset.dataset),
                     initial_model=state_digest(model),
                     torch_version=str(torch.__version__), cache_sha256=None)
    annotation = Path(args.root_dir) / args.dataset_name / 'data_captions.json'
    signature['annotations_sha256'] = file_digest(annotation)
    if args.sampler == 'balanced_mixed':
        _, meta = validate_cache(args.sampler_mining_cache, train_loader.dataset.dataset,
                                 state_digest(model.base_model))
        signature['cache_sha256'] = meta['sha256']
    model.cuda()
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    checkpointer = Checkpointer(model, optimizer, scheduler, str(output), True)
    evaluator = Evaluator(val_img, val_txt)
    start_epoch, saved = 1, {}
    resume_path = output / 'resume.pth'
    if args.resume:
        path = Path(args.resume_ckpt_file) if args.resume_ckpt_file else resume_path
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved.get('signature') != signature:
            raise ValueError('Cannot resume: model, data, configuration or cache changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'])
        start_epoch = saved['epoch'] + 1
        if saved.get('best_epoch') is not None and not (output / 'best.pth').exists():
            raise FileNotFoundError('Restore best.pth together with resume.pth')
    elif resume_path.exists() or (output / 'best.pth').exists():
        raise FileExistsError('Run exists: use --resume or choose a new output directory')
    save_train_configs(str(output), args)
    write_json(output / 'config.json', dict(args=vars(args), signature=signature))
    complete = do_train(start_epoch, args, model, train_loader, evaluator,
                        optimizer, scheduler, checkpointer,
                        resume_state=saved, signature=signature)
    del saved
    if not complete:
        logger.info('Paused at an epoch boundary. Save notebook output, then resume.')
        return
    logger.info('===================>start test')
    args.training = False
    test_img, test_txt, _ = build_dataloader(args)
    results = {}
    for name in ('best', 'last'):
        checkpoint = torch.load(output / f'{name}.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        evaluation = Evaluator(test_img, test_txt)
        evaluation.eval(model.eval())
        results[name] = evaluation.last_metrics
        write_json(output / f'{name}_test.json', results[name])
    write_json(output / 'result.json', dict(status='complete', completed_epochs=args.num_epoch,
               sampler=args.sampler, signature=signature, test=results,
               selection=f'best BGE+TSE R1 on {args.val_dataset} split'))


if __name__ == '__main__':
    main()
