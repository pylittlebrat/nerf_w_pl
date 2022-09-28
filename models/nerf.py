from re import T
from turtle import forward
import torch
from torch import nn

class PosEmbedding(nn.Module):
    """
    定义输入x编码为(x, sin(2^k x), cos(2^k x), ...)
    """
    def __init__(self,max_logscale,N_freqs,logscale=True) -> None:
        super().__init__()

        self.funcs = [torch.sin,torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0,max_logscale,N_freqs)
        else:
            self.freqs = torch.linspace(1,2**max_logscale,N_freqs)

    def forward(self,x):
        """
        Inputs:
        x:(B,3)

        Outputs:
        out:(B,6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

class NeRF(nn.Module):
    def __init__(self, typ,
                D=8,W=512,skips=[4],
                in_channels_xyz=63,in_channels_dir=27,
                encode_appearance=False,in_channels_a=48,
                encode_transient=False,in_channels_t=16,
                beta_min=0.03) -> None:
        """
        原始NeRF的参数:
        D:密度(sigma)编码器的层数
        W:每层隐藏单位的数量
        skips:在Dth层增加跳跃式连接
        In_channels_xyz: xyz的输入通道数量(默认为3+3*10*2=63)
        In_channels_dir:方向输入通道数(默认为3+3*4*2=27)
        In_channels_t: t的输入通道数

        NeRF-W的参数(只在精细模型中根据4.3节使用)——
        ——cf.论文的图3——
        encode_appearance:是否添加外观编码作为输入(NeRF-A)
        In_channels_a:外观嵌入维度。N ^(a)
        encode_transient:是否添加瞬态编码作为输入(NeRF-U)
        In_channels_t:瞬时嵌入维度。N^(tau)
        Beta_min:最小像素颜色方差
        """
        super().__init__()
        self.typ = typ # ?Fine网络和Coarse网络标签参数
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ=='coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min

        #xyz编码层
        for i in range(D):
            if i == 0:
                layer = nn.linear(in_channels_xyz,self.W)
            elif i in skips:
                layer = nn.linear(self.W+in_channels_xyz,self.W)
            else:
                layer = nn.linear(W,W)
            layer = nn.Sequential(layer,nn.LeakyReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)  # 每层网络命名，setattr(object, name, value)
        self.xyz_encoding_final = nn.Linear(self.W,self.W)

        # dir编码层
        self.dir_encoding = nn.Sequential(
            nn.Linear(self.W+in_channels_dir+self.in_channels_a,W//2),nn.LeakyReLU(True)
        )

        # original nerf output layers
        self.static_sigma = nn.Sequential(nn.Linear(self.W,1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(self.W//2,3),nn.Sigmoid())

        if self.encode_transient:
            # 瞬态物体编码层
            self.transient_encoding = nn.Sequential(
                                                    nn.Linear(self.W+in_channels_t,self.W//2),nn.LeakyReLU(True),
                                                    nn.Linear(self.W//2, self.W//2), nn.LeakyReLU(True),
                                                    nn.Linear(self.W//2, self.W//2), nn.LeakyReLU(True),
                                                    nn.Linear(self.W//2, self.W//2), nn.LeakyReLU(True)
            )
            self.transient_sigma = nn.Sequential(nn.Linear(self.W//2,1),nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(self.W//2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(self.W//2, 1), nn.Softplus()
            )
    
    def forward(self,x,sigma_only=False,output_transient=True):
        """
        将输入(xyz+dir)编码为rgb+sigma(尚未准备好渲染)。
        要渲染这条射线，请参见render .py

        输入:
        X:位置嵌入向量(+方向+外观+瞬态)
        Sigma_only:是否只推断sigma。
        Has_transient:是否推断瞬态组件。

        输出(连接):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            input_xyz = x

        elif output_transient:
            input_xyz,input_dir_a,input_t =  torch.split(x, [self.in_channels_xyz,
                                                        self.in_channels_dir+self.in_channels_a,
                                                        self.in_channels_t], dim=-1)

        else:
            input_xyz, input_dir_a = torch.split(x, [self.in_channels_xyz,
                                                self.in_channels_dir+self.in_channels_a], dim=-1)
        
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_) # (B, 1)

        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) # (B, 5)

        return torch.cat([static, transient], 1) # (B, 9)