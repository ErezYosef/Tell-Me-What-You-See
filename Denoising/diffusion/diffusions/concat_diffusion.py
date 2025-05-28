import guided_diffusion.diffusions as diffusions
import torch
from guided_diffusion import dist_util

BaseDiffusion = diffusions.get_diffusion('base_diffusion')

class BaseDiffusion_wrap(BaseDiffusion):

    def training_losses(self, model, x_start, t, model_kwargs=None):
        #print('start training losses step')
        if 'lq' not in model_kwargs:
            raise ValueError('Should be for base diffusion!')
        model_kwargs['low_res'] = model_kwargs['lq'] * 2 - 1  # concat the iamge to the diffusion input
        return super().training_losses(model, x_start, t, model_kwargs)

    @torch.no_grad()
    def p_sample_loop(self, model, shape, *args, **kwargs):
        #print('start p_sample_loop step')
        kwargs['model_kwargs']['low_res'] = kwargs['model_kwargs']['lq'] * 2 - 1
        kwargs['model_kwargs'].pop('lq')
        additional_args = {'low_res': kwargs['model_kwargs']['low_res']}
        shape = list(shape)

        shape[-1] = kwargs['model_kwargs']['low_res'].shape[-1]
        shape[-2] = kwargs['model_kwargs']['low_res'].shape[-2]
        # kwargs['diffusion_start_point'] = 3
        return super().p_sample_loop(model, shape, *args, **kwargs), additional_args

    def _adapt_kwargs_inputs_for_sampling(self, data_dict):
        # adapting sampling input args of 'p_sample_loop' (based on implementation).
        ret_dict = {}
        model_kwargs = {'lq': data_dict['lq'].to(dtype=torch.float32, device=dist_util.dev())}
        if 'clip_condition' in data_dict:
            model_kwargs['clip_condition'] = data_dict['clip_condition'].to(dtype=torch.float32, device=dist_util.dev())
        ret_dict['model_kwargs'] = model_kwargs
        ret_dict['noise'] = None
        #print('adapted')

        return ret_dict

