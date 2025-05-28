"""
Train a diffusion model on images.
"""
import sys
sys.path.append('..')
import torch
import argparse
from guided_diffusion import dist_util, logger
#from guided_diffusion.image_datasets import load_data
from guided_diffusion.datasets.sidd_raw_dataset import warp_DataLoader
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    all_args_to_dict,
)
from guided_diffusion.defaults_and_args import model_and_diffusion_defaults
#from guided_diffusion.train_util import TrainLoop
from train_util_wrap import TrainLoop_wrap as TrainLoop
from guided_diffusion.script_util import parse_yaml
from guided_diffusion.respace_diffusion import SpacedDiffusion
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule
from Denoising.diffusion.datasets import get_dataset # create_model_wrap

#from diffusion.diffusions.coldmix_diffusion import ColdMixDiffusion_wrap

from diffusions import get_model, get_diffusion, create_model_wrap_clean # create_model_wrap

def main():
    args = create_argparser().parse_args()
    non_default_args = get_non_default_args(args) # create_argparser().parse_args([])
    #print(non_default_args)
    args = parse_yaml(args, adapt_paths_to_machine=True, non_default_args=non_default_args)
    print(args)

    dist_util.setup_dist()
    logger.configure(args=args)

    logger.log(f'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))
    logger.log("creating model and diffusion...")
    #model, diffusion = create_model_and_diffusion(
    #    **args_to_dict(args, model_and_diffusion_defaults().keys()))
    print('pass1')
    model_class = get_model(args.model_type) #ConcatModel_wrappret_class(UNetModel) #ConcatModelConv
    model = create_model_wrap_clean(model_class=model_class, **all_args_to_dict(args))
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    diffusion_args = all_args_to_dict(args)
    diffusion_args['betas'] = betas
    diffusion_class = get_diffusion(args.diffusion_type) #diffusions.get_diffusion(args.diffusion_type)  # ColdMix or BaseDiffusion
    diffusion = SpacedDiffusion(diffusion_class, **diffusion_args)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"creating data loader... dir: {args.data_dir}")

    Dataset_class = get_dataset(args.dataset_type) #ConcatModel_wrappret_class(UNetModel) #ConcatModelCon
    train_ds = Dataset_class(main_path_dataset=args.main_data_path_train, mode='train', **diffusion_args)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                               collate_fn=None, drop_last=True)
    infinite_train_loader = warp_DataLoader(train_loader)

    #val_ds = Dataset_PairedImage(args.main_data_path, cropsize=args.image_size, phase='val', val_percent=args.val_percent)
    val_ds_args = dict(diffusion_args, trim_len=8, train_unlabeld=0) # copy and update the dict
    val_ds = Dataset_class(main_path_dataset=args.main_data_path_val, mode='val', **val_ds_args)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=args.num_workers)

    logger.log("training...")

    def check_params(model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN in parameter {name}")
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN in gradient of {name}")
    check_params(model)

    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_data=infinite_train_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        val_dataset=val_loader,
        batches_accumulate_grads=args.batches_accumulate_grads,
        test_dataset=None,
        islora=args.islora,
        lora_checkpoint= getattr(args, 'lora_checkpoint', None) if args.islora else None
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        config_file='image_train_config.yaml',
        format_strs='log,csv',
        wandb_tags=[],
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def get_non_default_args(args, default_args=None):
    import sys
    print('===', sys.argv)
    non_def_names = [a[2:] for a in sys.argv if a.startswith('--')]
    dict1 = args.__dict__
    #dict2 = default_args.__dict__
    non_def = {}
    for k in non_def_names:
        if k in dict1:
            non_def[k] = dict1[k] # override the future yaml loading by saving cmd line args
        else:
            print(f'Warning, arg: {k} in sys.argv, but not in argparser. Cant overided')
    print(non_def)
    return non_def

if __name__ == "__main__":
    main()
