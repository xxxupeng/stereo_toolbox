import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

class UpdateHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(UpdateHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class SigmoidUpdateHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(SigmoidUpdateHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return F.sigmoid(self.conv2(self.relu(self.conv1(x))))
    
class ScaleShiftUpdateHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(ScaleShiftUpdateHead, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.scaler = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),            
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        B = x.shape[0]
        backbone = self.conv2(self.relu(self.conv1(x)))
        return self.scaler(backbone).reshape(B, self.output_dim, 1, 1)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)

        h = (1-z) * h + z * q
        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        self.corr_levels = args.corr_levels
        self.corr_radius = args.corr_radius
        self.encoder_output_dim = args.encoder_output_dim

        cor_planes = self.corr_levels * (2*self.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self._conv = nn.Conv2d(64+64+64, 128-2, 3, padding=1)

    def forward(self, flow, corr, corr_mono):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1(corr_mono))
        cor_mono = F.relu(self.convc2(cor_mono))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, cor_mono, flo], dim=1)
        out = F.relu(self._conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicConfidenceAwareMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicConfidenceAwareMotionEncoder, self).__init__()
        self.args = args
        self.corr_levels = args.corr_levels
        self.corr_radius = args.corr_radius
        self.encoder_output_dim = args.encoder_output_dim

        cor_planes = self.corr_levels * (2*self.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convcf1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convcf2 = nn.Conv2d(64, 64, 3, padding=1)
        self._conv_with_conf = nn.Conv2d(64+64+64+64, 128-3, 3, padding=1)

    def forward(self, flow, flow_conf, corr, corr_mono):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1(corr_mono))
        cor_mono = F.relu(self.convc2(cor_mono))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        flo_conf = F.relu(self.convcf1(flow_conf))
        flo_conf = F.relu(self.convcf2(flo_conf))

        cor_flo = torch.cat([cor, cor_mono, flo, flo_conf], dim=1)
        out = F.relu(self._conv_with_conf(cor_flo))
        return torch.cat([out, flow, flow_conf], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[], predict_confidence=True):
        super().__init__()
        self.args = args
        self.predict_confidence = predict_confidence

        if self.predict_confidence:
            self.encoder = BasicConfidenceAwareMotionEncoder(args)
        else:
            self.encoder = BasicMotionEncoder(args)

        encoder_output_dim = args.encoder_output_dim
        self.n_gru_layers = args.n_gru_layers
        self.n_downsample = args.n_downsample

        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (self.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (self.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])

        self.flow_head = UpdateHead(hidden_dims[2], hidden_dim=256, output_dim=2)

        if self.predict_confidence:
            self.conf_head = SigmoidUpdateHead(hidden_dims[2], hidden_dim=256, output_dim=1)

        factor = 2**self.n_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))

    def forward(self, net, inp, corr=None, corr_mono=None, flow=None, flow_conf=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            
            if self.predict_confidence:
                motion_features = self.encoder(flow, flow_conf, corr, corr_mono)
            else:
                motion_features = self.encoder(flow, corr, corr_mono)

            if self.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])

        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])

        if self.predict_confidence:
            delta_confidence = self.conf_head(net[0])
            return net, mask, delta_flow, delta_confidence

        return net, mask, delta_flow

class BasicMultiUpdateScalerBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicConfidenceAwareMotionEncoder(args)
        encoder_output_dim = 128
        self.n_gru_layers = 3
        self.n_downsample = 2

        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (self.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (self.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])

        self.lscale_head = SigmoidUpdateHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        self.conf_head = SigmoidUpdateHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        self.gscale_gshift_head = ScaleShiftUpdateHead(hidden_dims[2], hidden_dim=256, output_dim=2)

    def forward(self, net, inp, corr=None, flow=None, flow_conf=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(flow, flow_conf, corr)
            if self.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_gscale_gshift = self.gscale_gshift_head(net[0])
        delta_gscale, delta_gshift = delta_gscale_gshift[:,0:1], delta_gscale_gshift[:,1:2]
        delta_confidence = self.conf_head(net[0])
        delta_lscale = self.lscale_head(net[0])

        return net, delta_lscale, delta_gscale, delta_gshift, delta_confidence
