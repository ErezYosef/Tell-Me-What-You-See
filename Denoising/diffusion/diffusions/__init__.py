from . import concat_diffusion
from . import concat_model
from . import conditioned_model
#from . import conditioned_model
import guided_diffusion.diffusions.get_diffusion
from guided_diffusion.script_util import create_model_new
from guided_diffusion.unet import UNetModel
from guided_diffusion.unet_other import UNetModelConv
from guided_diffusion.glide.text2im_model import UnetConditionModel

import torch

def get_diffusion(name):
    if name == 'base':
        return concat_diffusion.BaseDiffusion_wrap
    elif name == 'condfull':
        raise NotImplementedError#return concat_diffusion.BaseDiffusion_condition
    else:
        print(f'Warning: diffusion class {name} is missing in flatnet > searching in guided_diffusion package..')
        return guided_diffusion.diffusions.get_diffusion(name)

def get_model(name):
    if name == 'concat_unetconv':
        return concat_model.ConcatModel_wrap(UNetModelConv)
    if name == 'concat_unet':
        return concat_model.ConcatModel_wrap(UNetModel)
    elif name == 'concat_condition':
        return concat_model.ConcatModel_wrap(UnetConditionModel)
    elif name == 'concat_condition_nulllabel':
        return concat_model.ConcatModel_wrap(conditioned_model.UnetConditionModel_nulllabel)
    elif name == 'unetconv':
        return UNetModelConv
    else:
        print(f'Warning: model {name} is missing in flatnet > error')
        return UNetModelConv


def create_model_wrap_clean(model_class, **kwargs):
    model = create_model_new(model_class=model_class, **kwargs)
    if kwargs['diffusion_type'] == 'soft' or kwargs['diffusion_type'] == 'condfull':
        from basicsr.models.archs.restormer_arch import Restormer
        yaml_file = 'Options/RealDenoising_Restormer.yml'
        import yaml
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
        s = x['network_g'].pop('type')

        model_restoration = Restormer(**x['network_g'])
        checkpoint = torch.load('pretrained_models/real_denoising.pth')
        model_restoration.load_state_dict(checkpoint['params'])
        model.restormer = model_restoration

        print('init DVIR model - added to model')
    if kwargs['diffusion_type'] == 'dvir_restormer_trained_freeze':
        from basicsr.models.archs.restormer_arch import Restormer
        yaml_file = 'Options/RealDenoisingRaw_Restormer.yml'
        import yaml
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
        s = x['network_g'].pop('type')

        model_restoration = Restormer(**x['network_g'])
        checkpoint = torch.load('pretrained_models/net_g_230000.pth')
        model_restoration.load_state_dict(checkpoint['params'])
        model.restormer = model_restoration
        pass
    if kwargs['diffusion_type'] == 'dvir_unet':
        args2 = kwargs.copy()
        args2['input_channels'] = 4
        model_restoration = create_model_new(model_class=conditioned_model.UNetModelConv0, **args2)
        model.restormer = model_restoration
        print('init DVIR model - added !! UNetModelConv !! to model')

    return model