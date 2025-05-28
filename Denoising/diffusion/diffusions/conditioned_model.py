import guided_diffusion.diffusions as diffusions
import torch
from guided_diffusion import dist_util

#from . import get_diffusion
from guided_diffusion.unet import timestep_embedding
pass
from guided_diffusion.unet_other import UNetModelConv

from guided_diffusion.glide.text2im_model import UnetConditionModel

class UnetConditionModel_nulllabel(UnetConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.null_label = torch.nn.Parameter(torch.randn(self.xf_width)) # self.xf_width

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        def check_and_update_shape(name):
            if state_dict[name].nelement() == getattr(self, name).nelement() \
                    and state_dict[name].shape != getattr(self, name).shape:
                print('reshaping state_dict for null_label')
                state_dict[name] = state_dict[name].view(*getattr(self, name).shape)

        print('DO LOAD STATEDICT', state_dict['null_label'].numel() == self.null_label.numel(), state_dict['null_label'].numel(), self.null_label.numel())
        # if state_dict['null_label'].nelement() == self.null_label.nelement():
        #     state_dict['null_label'] = state_dict['null_label'].view(*self.null_label.shape)
        #     print('reshaping state_dict for null_label')
        check_and_update_shape('null_label')
        super()._load_from_state_dict(state_dict, *args, **kwargs)

    def forward(self, x, timesteps=None, clip_condition=None, *args, **kwargs):
        #print(clip_condition.shape, clip_condition.dim)
        # if clip_condition.dim() == 1:
        #     #print('ENTER TO IF ', clip_condition.dim, clip_condition.shape)
        #     bs = clip_condition.shape[0]
        #     clip_condition = self.null_label.expand(bs, -1)
        #     #print(clip_condition.shape)
        for i in range(clip_condition.shape[0]):
            if clip_condition[i].sum() == 0:
                #print('ENTER TO IF ', clip_condition.dim, clip_condition.shape)
                #bs = clip_condition.shape[0]
                #print('replaced_embnull')
                clip_condition[i] = self.null_label
                #print(clip_condition.shape)

        return super().forward(x, timesteps=timesteps, clip_condition=clip_condition, *args, **kwargs)


class UNetModelConv0(UNetModelConv):
    def forward(self, x, timesteps=None, *args, **kwargs):
        assert timesteps is None
        bs = x.shape[0]
        timesteps = torch.zeros(bs).to(x) # device and dtype

        return super().forward(x, timesteps=timesteps, *args, **kwargs)


'''
def ConcatCondModel_wrap(model_class):
    class ConcatModel_wrapper(model_class):
        def forward(self, x, timesteps, low_res=None, y=None, **kwargs):
            x = torch.cat([x, low_res], dim=1)
            if self.num_classes is not None and y is None:
                print(f"not specified y if the model is class-conditional, {self.num_classes}")

            hs = []
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            if self.num_classes is not None:
                #assert y.shape[0] == x.shape[0], f'{y.shape} != {x.shape}'
                emb = emb #+ self.label_emb(y) # todo think

            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb)
                hs.append(h)
            h = self.middle_block(h, emb)
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb)
            h = h.type(x.dtype)
            return self.out(h)

    return ConcatModel_wrapper

#from flatnet_sep.FlatNet_separable import inconv, down, up, outconv
import torch.nn as nn
import torch.nn.functional as F

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class double_conv_cond(nn.Module):
    def __init__(self, channels, out_channels=None, emb_channels=32*4, stride=1):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_channels=channels, num_groups=2, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, stride=stride, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels))

        self.out_layers = nn.Sequential(
            #GroupNorm32(num_channels=self.out_channels, num_groups=32, swish=1),
            nn.GroupNorm(num_channels=self.out_channels, num_groups=2, eps=1e-5),
            nn.SiLU(),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
        )

        if self.out_channels == channels and stride==1:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, stride=stride, padding=1)

    def forward(self, x, emb):
        #print('x', x.shape)
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        #print('xhs', x.shape, h.shape, self.skip_connection.weight.shape)
        return self.skip_connection(x) + h

class double_conv(nn.Module):
    # (conv => BN => ReLU) * 2

    def __init__(self, in_ch, out_ch, emb_channels=32*4):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.99),
            nn.ReLU(inplace=True)
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(emb_channels, out_ch)))

    def forward(self, x, emb=None):
        x = self.conv[:4](x)
        if emb is not None:
            emb_out = self.emb_layers(emb).type(x.dtype)
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out[..., None]
            x = x + emb_out
        x = self.conv[4:](x)
        return x


class double_conv2(nn.Module):
    #(conv => BN => ReLU) * 2

    def __init__(self, in_ch, out_ch, emb_channels=32*4):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.99),
            nn.ReLU(inplace=True)
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(emb_channels, out_ch)))

    def forward(self, x, emb=None):
        x = self.conv[:4](x)
        if emb is not None:
            emb_out = self.emb_layers(emb).type(x.dtype)
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out[..., None]
            x = x + emb_out
        x = self.conv[4:](x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x, emb=None):
        x = self.conv(x, emb)
        return x

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self:
            input = module(*input)
        return input

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = mySequential(
            double_conv2(in_ch, out_ch)
        )

    def forward(self, x, emb=None):
        x = self.mpconv(x, emb)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2, emb=None):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, emb)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x



class FlatNet_orig(nn.Module):
    def __init__(self, n_channels=4):
        super().__init__()
        self.inc = inconv(n_channels, 128)
        self.down1 = down(128, 256)
        self.down2 = down(256, 512)
        self.down3 = down(512, 1024)
        self.down4 = down(1024, 1024)
        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 128)
        self.outc = outconv(128, 3)
        self.PhiL = nn.Parameter(torch.randn(500, 256, 1))
        self.PhiR = nn.Parameter(torch.randn(620, 256, 1))
        self.bn = nn.BatchNorm2d(4, momentum=0.99)

        self.cond_projection = FullyConnected(512, 32 * 4, layers_num=2, act_last=False)

    def forward(self, Xinp, clip_condition=None):
        X0 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:, 0, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                                       self.PhiL[:, :, 0]).permute(0, 2, 1).unsqueeze(3))
        X11 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:, 1, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                                        self.PhiL[:, :, 0]).permute(0, 2, 1).unsqueeze(3))
        X12 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:, 2, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                                        self.PhiL[:, :, 0]).permute(0, 2, 1).unsqueeze(3))
        X2 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:, 3, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                                       self.PhiL[:, :, 0]).permute(0, 2, 1).unsqueeze(3))
        Xout = torch.cat((X2, X12, X11, X0), 3)
        x = Xout.permute(0, 3, 1, 2)
        x = self.bn(x)
        xf_proj = self.cond_projection(clip_condition)
        x1 = self.inc(x, xf_proj)
        x2 = self.down1(x1, xf_proj)
        x3 = self.down2(x2, xf_proj)
        x4 = self.down3(x3, xf_proj)
        x5 = self.down4(x4, xf_proj)
        x = self.up1(x5, x4, xf_proj)
        x = self.up2(x, x3, xf_proj)
        x = self.up3(x, x2, xf_proj)
        x = self.up4(x, x1, xf_proj)
        x = self.outc(x)

        return torch.sigmoid(x)

class FullyConnected(nn.Module):
    def __init__(self, in_feat, out_feat, layers_num=1, act_last=True):
        super(FullyConnected, self).__init__()
        layers_list = []
        assert layers_num > 0
        for i in range(layers_num):
            layers_list.append(nn.Linear(in_feat, out_feat))
            in_feat = out_feat
            act = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            layers_list.append(act)
        if not act_last:
            layers_list = layers_list[:-1]
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layers(x)

class FlatNet_cond(nn.Module):
    def __init__(self, n_channels=4):
        super().__init__()
        self.inconv = double_conv_cond(n_channels, 128)
        self.down1 = double_conv_cond(128, 256, stride=2)
        self.down2 = double_conv_cond(256, 512, stride=2)
        self.down3 = double_conv_cond(512, 1024, stride=2)
        self.down4 = double_conv_cond(1024, 1024, stride=2)
        self.up1 = up(2048, 512, double_cblock=double_conv_cond)
        self.up2 = up(1024, 256, double_cblock=double_conv_cond)
        self.up3 = up(512, 128, double_cblock=double_conv_cond)
        self.up4 = up(256, 128, double_cblock=double_conv_cond)
        self.outc = nn.Conv2d(128, 3, 3, padding=1)
        self.PhiL = nn.Parameter(torch.randn(500, 256, 1))
        self.PhiR = nn.Parameter(torch.randn(620, 256, 1))
        self.bn = nn.BatchNorm2d(4, momentum=0.99)

        self.cond_projection = FullyConnected(512, 32 * 4, layers_num=2, act_last=False)

    def forward(self, Xinp, clip_condition=None):
        X0 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:, 0, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                                       self.PhiL[:, :, 0]).permute(0, 2, 1).unsqueeze(3))
        X11 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:, 1, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                                        self.PhiL[:, :, 0]).permute(0, 2, 1).unsqueeze(3))
        X12 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:, 2, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                                        self.PhiL[:, :, 0]).permute(0, 2, 1).unsqueeze(3))
        X2 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:, 3, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                                       self.PhiL[:, :, 0]).permute(0, 2, 1).unsqueeze(3))
        Xout = torch.cat((X2, X12, X11, X0), 3)
        x = Xout.permute(0, 3, 1, 2)
        x = self.bn(x)
        xf_proj = self.cond_projection(clip_condition)
        x1 = self.inconv(x, xf_proj)
        x2 = self.down1(x1, xf_proj)
        x3 = self.down2(x2, xf_proj)
        x4 = self.down3(x3, xf_proj)
        x5 = self.down4(x4, xf_proj)
        x = self.up1(x5, x4, xf_proj)
        x = self.up2(x, x3, xf_proj)
        x = self.up3(x, x2, xf_proj)
        x = self.up4(x, x1, xf_proj)
        x = self.outc(x)

        return x

'''
pass