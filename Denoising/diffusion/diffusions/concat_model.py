import torch
from guided_diffusion.diffusions.lora import lora_model_for_diffusion

def ConcatModel_wrap(model_class):
    class ConcatModel_wrapper(lora_model_for_diffusion, model_class):
        """
        A wrapper that performs concatenation.

        Expects an extra kwarg `low_res` to condition on a low-resolution image.
        """
        # def __init__(self, image_size, in_channels, *args, **kwargs):
        #     #print(image_size, in_channels)
        #     super().__init__(image_size, in_channels * 2, *args, **kwargs)

        def forward(self, x, timesteps, low_res=None, **kwargs):
            #_, _, new_height, new_width = x.shape
            # upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
            x = torch.cat([x, low_res], dim=1)
            #print('sshape', x.shape)
            return super().forward(x, timesteps, **kwargs)
    return ConcatModel_wrapper
