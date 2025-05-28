import os
import torch
import torch.distributed as dist
#import sys
import lpips
#sys.path.append('..')

from guided_diffusion.glide.ssim import ssim

from guided_diffusion import dist_util, logger
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.saving_imgs_utils import save_img,tensor2img
from guided_diffusion.process_raw2rgb_torch import process
from guided_diffusion.diffusions.lora import inject_trainable_lora_with_group
from functools import partial


class TrainLoop_wrap(TrainLoop):
    def _load_and_sync_parameters(self):
        #print('==== islora', self.model.islora)
        super()._load_and_sync_parameters()
        if not self.kwargs.get('islora', False):
            return
        # else:
        lora_module_list = inject_trainable_lora_with_group(self.model, exclude_names=('restormer',), r=4)
        self.model.islora = True
        self.model.lora_module_list = lora_module_list
        resume_checkpoint_lora = self.kwargs.get('lora_checkpoint', None)
        if resume_checkpoint_lora:
            print('=== resume_checkpoint LORA from: ', resume_checkpoint_lora, self.resume_step)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint_lora}...")
                self.model.lora_module_list.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint_lora, map_location=dist_util.dev()
                    )
                )
        print('LORA WAS APPLIED; TOTAL PARAMS LORA:')
        print(sum(p.numel() for p in self.model.parameters()))
        dist_util.sync_params(self.model.parameters())


    def run_loop(self):
        while (not self.lr_anneal_steps or self.totstep < self.lr_anneal_steps):
            self.run_step()
            if self.totstep % self.save_interval == 0:
                # override files and save disk space in factor 5
                self.save(file_identifier=None if self._if_save_with_step_num() else 'latest')
                logger.log("sampling images...")
                #print('step:', self.totstep, self.step)
                self.validation_sample(self.val_dataset, only_first_batch=False, call_id=0)
                logger.log("sampling completed")
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    print('recieved DIFFUSION_TRAINING_TEST indication')
                    return
            if self.totstep % self.log_interval == 0:
                logger.dumpkvs()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.totstep - 1) % self.save_interval != 0:
            self.save()
    def _if_save_with_step_num(self):
        # Save data with step number (and not 'latest' overrided) every 25000 / or if lora always save numberd
        return self.totstep % 25000 == 0 and self.totstep>0 or getattr(self.model, 'islora', False)

    def forward_backward(self, batch, cond, use_device=None):
        return super().forward_backward(batch, cond, use_device=None) # torch.device('cpu')

    def validation_sample(self, data_to_sample=None, num_samples=8, only_first_batch=True, call_id=0, save_all=False,
                          update_logger_for_sample=False, log_images_wandb=True, post_tag_folder=''):
        self.model.eval()
        if data_to_sample is None:
            data_to_sample = self.val_dataset
        image_size = self.model.image_size
        # Local setup
        clip_denoised = True

        all_images = []
        mse = torch.nn.MSELoss()
        if not hasattr(self, 'loss_fn') or self.loss_fn is None:
            self.loss_fn = lpips.LPIPS(net='alex').to(dist_util.dev())
        loss_fn = self.loss_fn
        mse_loss, ssim_loss = 0, 0
        lpips_loss = 0
        #batch_counter = 0
        samples_counter = 0
        total_saved_samples = 0
        #raw2rgb_process11 = partial(process, min_max=(-1,1))
        #                         val_scores = eval(nets[i], val_loader, device, get_vgg=True)
        from tqdm import tqdm
        #for sample_condition_data in tqdm(data_to_sample, desc='Validation round', unit='batch'):
        #print('lenlen', len(data_to_sample))
        for batch_counter, sample_condition_data in enumerate(data_to_sample):
            print(sample_condition_data[0].shape)
            model_kwargs = {}
            batch_size = sample_condition_data[0].shape[0]
            sample_fn = self.diffusion.p_sample_loop # if not args.use_ddim else self.diffusion.ddim_sample_loop)
            gt_imgs, data_dict = sample_condition_data

            gt_imgs = gt_imgs.to(dtype=torch.float32, device=dist_util.dev())
            sample, x_T_end = sample_fn(
                self.model,
                (batch_size, 4, image_size, image_size),
                clip_denoised=clip_denoised,
                **self.diffusion._adapt_kwargs_inputs_for_sampling(data_dict)
            )
            # Copy from image sample code:
            sample_cp = sample.clone()
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            #all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

            mse_loss += mse(sample_cp, gt_imgs) * sample_cp.shape[0] /4 # due to dynamic range
            sample_cp_rgb = process(sample_cp, min_max=(-1,1))
            gt_imgs_rgb = process(gt_imgs, min_max=(-1,1))
            ssim_loss += ssim(sample_cp_rgb, gt_imgs_rgb, val_range=1) * sample_cp.shape[0]
            if gt_imgs_rgb.shape[1] == 3:
                lpips_loss += loss_fn(sample_cp_rgb, gt_imgs_rgb, normalize=True).mean() * sample_cp.shape[0]
            #x_T_end = (x_T_end, ) if not isinstance(x_T_end, tuple) else x_T_end
            assert isinstance(x_T_end, dict), 'x_T_end (returned from p_sample_loop) should be a dict!'

            #if batch_counter == 0:
            if log_images_wandb:
                if self.step == 0:  # was "==0" but changed to support resume train from 999999
                    logger.get_logger().logimage(f'img{call_id}_input0', gt_imgs.to('cpu'))
                for k,v in x_T_end.items():
                    if k == 'low_res':
                        if self.step == 0:
                            logger.get_logger().logimage(f'img{call_id}_{k}', v)
                    else:
                        logger.get_logger().logimage(f"img{call_id}_{k}", v)
                #print(f'logging to img{call_id}_xT at {self.step}, total {len(x_T_end)}')
                logger.get_logger().logimage(f'img{call_id}_samples0', sample_cp)
                if update_logger_for_sample:  # in code: image_sample.py
                    logger.dumpkvs()  # dump the results
                    self.step += 1  # update next step
                    self.log_step()  # log the step for next loop if available



            if save_all:
                dir_to_save = os.path.join(logger.get_dir(), 'save'+post_tag_folder)
                dir_to_init = os.path.join(logger.get_dir(), 'save'+post_tag_folder)
                dir_to_gt = os.path.join(logger.get_dir(), 'save'+post_tag_folder)
                if not os.path.exists(dir_to_save):
                    os.mkdir(dir_to_save)
                for i in range(sample_cp.shape[0]):
                    img = sample_cp[i].clone()
                    img_path = os.path.join(dir_to_save, f'sample{total_saved_samples:03d}.png')
                    #print(img_path, os.path.exists(dir_to_save))
                    save_img(tensor2img(img), img_path)
                    torch.save(img, img_path.replace('.png', '.pt'))
                    gt_path = os.path.join(dir_to_gt, f'gt{total_saved_samples:03d}.png')
                    save_img(tensor2img(gt_imgs[i]), gt_path)
                    torch.save(gt_imgs[i].clone(), gt_path.replace('.png', '.pt'))
                    for k, v in x_T_end.items():
                        dict_item_path = os.path.join(dir_to_init, f"sample{total_saved_samples:03d}_{k}.png")
                        save_img(tensor2img(v[i]), dict_item_path)
                        torch.save(v[i].clone(), dict_item_path.replace('.png', '.pt'))

                    total_saved_samples += 1


            samples_counter += sample.shape[0]
            print(samples_counter, end='\r')
            if batch_counter == 0 and only_first_batch:
                break

        avg_tot_loss = mse_loss / samples_counter
        avg_tot_ssim = ssim_loss / samples_counter
        avg_tot_lpips = lpips_loss / samples_counter

        logger.logkv(f'PSNR_{call_id}', -10 * torch.log10(avg_tot_loss))
        logger.logkv(f'SSIM_{call_id}', avg_tot_ssim)
        logger.logkv(f'LPIPS_{call_id}', avg_tot_lpips)
        logger.logkv(f'samples_counter', samples_counter)

        dist.barrier()
        #logger.log("sampling complete")
        self.model.train()
        print(self.totstep, end='\r')
