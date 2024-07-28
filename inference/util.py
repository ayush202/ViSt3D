from einops import rearrange,reduce
import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model1_part0 = nn.Sequential(
            nn.ReflectionPad3d(1),  # c t+2 h+2 w +2
            nn.Conv3d(512, 256, (3, 3, 3)),  # c t h w
            nn.ReLU(),
            nn.ReflectionPad3d(1),  # c t+2 h+2 w +2
            nn.Conv3d(256, 256, (3, 3, 3)),  # c t h w
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')  # c/2 t h w -> c/2 2t 2h 2w
        )

        self.model1_part1 = nn.Sequential(
            #Concatenation of li[2]
            nn.ReflectionPad3d(1),  # c 2t+2 2h+2 2w +2
            nn.Conv3d(512, 256, (3, 3, 3)),  # c 2t 2h 2w
            nn.ReLU(),
            nn.ReflectionPad3d(1),  # c 2t+2 2h+2 2w +2
            nn.Conv3d(256, 256, (3, 3, 3)), # c/2 2t 2h 2w
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),  # c/2 2t 2h 2w -> c/2 4t 4h 4w
        )
        self.model1_part2 = nn.Sequential(
            #Concatenate li[1] here
            nn.ReflectionPad3d(1),  # c 4t+2 4h+2 4w +2
            nn.Conv3d(384, 384, (3, 3, 3)),  # c 4t 4h 4w
            nn.ReLU(),
            nn.ReflectionPad3d(1),  # c 4t+2 4h+2 4w+2
            nn.Conv3d(384, 256, (3, 3, 3)),  # c/2 4t 4h 4w
            nn.ReLU(),
            nn.ReflectionPad3d(1),  # c/2 4t+2 4h+2 4w+2
            nn.Conv3d(256, 256, (3, 3, 3)),  # c/2 4t 4h 4w
            nn.ReLU(),
            nn.ReflectionPad3d(1),
            nn.Conv3d(256, 128, (3, 3, 3)),  # c/2 ... -> c/4 ...
            nn.ReLU(),
            nn.ReflectionPad3d(2),  # c/4 4t+4 4h+4 4w+4
            nn.Conv3d(128, 128, (3, 3, 3)),  # c/4 4t+2 4h+2 4w+2
            nn.ReLU(),
            nn.Conv3d(128, 64, (3, 3, 3)),  # c/4 ... -> c/8 4t 4h 4w
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),  # c/8 4t 4h 4w -> c/4 4t 8h 8w
        )
        self.model1_part3 = nn.Sequential(
            #Concatenate li[0] here
            nn.ReflectionPad3d(1),  # c/4 4t+2 4h+2 4w+2
            nn.Conv3d(128, 128, (3, 3, 3)),  # c/4 4t 4h 4w
            nn.ReLU(),
            nn.ReflectionPad3d(1),  # c/4 4t+2 4h+2 4w+2
            nn.Conv3d(128, 64, (3, 3, 3)),  # c/8 4t 4h 4w
            nn.ReLU(),
            nn.ReflectionPad3d(1),  # c/4 4t+2 4h+2 4w+2
            nn.Conv3d(64, 3, (3, 3, 3))  # 3 4t 4h 4w
            )

    def forward(self, data,concatenated_tensor, device):
        feat = concatenated_tensor.to(device)
        self.model1_part0.to(device)
        feat = self.model1_part0(feat)
        feat = torch.concat((feat , data[2].to(device)), dim=1)
        self.model1_part1.to(device)
        feat = self.model1_part1(feat)
        feat = torch.concat((feat, data[1].to(device)), dim=1)
        self.model1_part2.to(device)
        feat = self.model1_part2(feat)
        self.model1_part3.to(device)
        feat = torch.concat((feat, data[0].to(device)), dim=1)
        return self.model1_part3(feat)

c3d_model = nn.Sequential(
    nn.Conv3d(3, 3, (1, 1, 1)),
    nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0, dilation=1, ceil_mode=False),
    nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1, ceil_mode=False),
    nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
    nn.ReLU(),
    nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1, ceil_mode=False),
    nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
    nn.ReLU()
)

class disentangle(nn.Module):
    def __init__(self):
        super(disentangle, self).__init__()
        self.model = nn.Sequential(
        nn.Conv3d(512, 512, (3, 3, 3), (1, 1, 1), padding = (1, 1, 1)),
        nn.ReLU(),
        nn.Conv3d(512, 512, (3, 3, 3), (1, 1, 1), padding = (1, 1, 1)),
        nn.ReLU(),
        nn.Conv3d(512, 512, (3, 3, 3), (1, 1, 1), padding = (1, 1, 1)),
        nn.ReLU(),
        nn.Conv3d(512, 512, (3, 3, 3), (1, 1, 1), padding = (1, 1, 1)),
        nn.ReLU(),
        nn.Conv3d(512, 512, (1, 1, 1), (1, 1, 1), padding = (0, 0, 0)),
        nn.ReLU(),
        )

    def forward(self, data, device):
        data = data.to(device)
        self.model.to(device)
        return self.model(data)

class disentangle1(nn.Module):
    def __init__(self):
        super(disentangle1, self).__init__()
        self.model = nn.Sequential(
        nn.Conv3d(256, 256, (3, 3, 3), (1, 1, 1), padding = (1, 1, 1)),
        nn.ReLU(),
        nn.Conv3d(256, 256, (3, 3, 3), (1, 1, 1), padding = (1, 1, 1)),
        nn.ReLU(),
        nn.Conv3d(256, 256, (3, 3, 3), (1, 1, 1), padding = (1, 1, 1)),
        nn.ReLU(),
        nn.Conv3d(256, 256, (1, 1, 1), (1, 1, 1), padding = (0, 0, 0)),
        nn.ReLU(),
        )

    def forward(self, data, device):
        data = data.to(device)
        self.model.to(device)
        return self.model(data)

class disentangle2(nn.Module):
    def __init__(self):
        super(disentangle2, self).__init__()
        self.model = nn.Sequential(
        nn.Conv3d(128, 128, (3, 3, 3), (1, 1, 1), padding = (1, 1, 1)),
        nn.ReLU(),
        nn.Conv3d(128, 128, (3, 3, 3), (1, 1, 1), padding = (1, 1, 1)),
        nn.ReLU(),
        nn.Conv3d(128, 128, (1, 1, 1), (1, 1, 1), padding = (0, 0, 0)),
        nn.ReLU(),
        )

    def forward(self, data, device):
        data = data.to(device)
        self.model.to(device)
        return self.model(data)

class disentangle3(nn.Module):
    def __init__(self):
        super(disentangle3, self).__init__()
        self.model = nn.Sequential(
        nn.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), padding = (1, 1, 1)),
        nn.ReLU(),
        nn.Conv3d(64, 64, (1, 1, 1), (1, 1, 1), padding = (0, 0, 0)),
        nn.ReLU(),
        )

    def forward(self, data, device):
        data = data.to(device)
        self.model.to(device)
        return self.model(data)

class entangle(nn.Module):
    def __init__(self):
        super(entangle, self).__init__()
        self.model = nn.Sequential(
        nn.Conv3d(1024, 512, (1, 1, 1), (1, 1, 1), padding = (0, 0, 0)),
        nn.ReLU()
        )
        val = torch.zeros((512,1024,1,1,1))

        for i in range(len(val)):
            val[i][i] = 0.3
            val[i][i+512] = 0.7

        self.model[0].weight = nn.Parameter(val)
        self.model[0].bias = nn.Parameter(torch.zeros(512))



    def forward(self, data, device):
        data = data.to(device)
        self.model.to(device)
        return self.model(data)

def calc_mean_std_3D(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 5)

    feat_var = rearrange(feat,'n c t h w -> n c (t h w)').var(dim=2) + eps
    #feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = rearrange(feat_var.sqrt(),'n c -> n c 1 1 1')
    #feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = reduce(rearrange(feat,'n c t h w -> n c (t h w)'),'n c thw -> n c 1 1 1','mean')
    #feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1) * lambda_style

    return feat_mean, feat_std

def adaptive_instance_normalization1_3D(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()

    style_mean, style_std = calc_mean_std_3D(style_feat)

    content_mean, content_std = calc_mean_std_3D(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)