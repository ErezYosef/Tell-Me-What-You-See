"""
Train a diffusion model on images.
"""
import sys
sys.path.append('..')
import torch
from guided_diffusion import dist_util, logger
#from guided_diffusion.image_datasets import load_data
from guided_diffusion.datasets.sidd_raw_dataset import warp_DataLoader
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    all_args_to_dict,
)
#from guided_diffusion.train_util import TrainLoop
from train_util_wrap import TrainLoop_wrap as TrainLoop
from guided_diffusion.script_util import parse_yaml
from guided_diffusion.respace_diffusion import SpacedDiffusion
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule

#from diffusion.diffusions.coldmix_diffusion import ColdMixDiffusion_wrap
from image_train import create_argparser
from guided_diffusion.script_util import load_folder_path_parse
from diffusions import get_model, get_diffusion, create_model_wrap_clean # create_model_wrap
from datasets import get_dataset # create_model_wrap
from image_train import get_non_default_args

def main():
    args = create_argparser().parse_args()
    non_default_args = get_non_default_args(args)
    args = parse_yaml(args, non_default_args=non_default_args)

    dist_util.setup_dist()
    loaded_folder_name = load_folder_path_parse(args)
    args.resume_checkpoint = args.model_path
    print(args)
    logger.configure(args=args, loaded_folder_name=loaded_folder_name)

    logger.log(f'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))
    logger.log("creating model and diffusion...")
    #model, diffusion = create_model_and_diffusion(
    #    **args_to_dict(args, model_and_diffusion_defaults().keys()))
    print('pass1')
    # if args.set_seed is not None:
    #     torch.manual_seed(args.set_seed)
    #     print(f'seed sets to : {args.set_seed}')
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

    #from dataset_sidd import Dataset_PairedImage_crops_less_clip as Dataset_class
    # last change: >> from Denoising.diffusion.datasets.dataset_real import Dataset_Realcam as Dataset_class
    Dataset_class = get_dataset(args.dataset_type) #ConcatModel_wrappret_class(UNetModel) #ConcatModelConv

    #train_ds = Dataset_PairedImage(args.main_data_path, cropsize=args.image_size, random_crop=True, phase='train', val_percent=args.val_percent)
    train_ds = Dataset_class(main_path_dataset=args.main_data_path_train, mode='train', **diffusion_args)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                               collate_fn=None, drop_last=True)
    infinite_train_loader = warp_DataLoader(train_loader)

    #val_ds = Dataset_PairedImage(args.main_data_path, cropsize=args.image_size, phase='val', val_percent=args.val_percent)
    val_ds = Dataset_class(main_path_dataset=args.main_data_path_val, mode='val', **diffusion_args)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, batch_sampler=getattr(val_ds, 'batch_loader', None))

    logger.log("training...")
    train_loop = TrainLoop(
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
        islora=getattr(args, 'islora', False),
        lora_checkpoint=getattr(args, 'lora_checkpoint', None) if getattr(args, 'islora', False) else None

    )
    train_loop.log_step()
    for i in range(args.ntimes):
        train_loop.validation_sample(only_first_batch=False, save_all=args.save_all_samples,
                                     update_logger_for_sample=False, log_images_wandb=False, post_tag_folder=f'{i}')
    logger.dumpkvs()


if __name__ == "__main__":
    main()
