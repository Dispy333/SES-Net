class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        # Sobel 核 (3x3)
        sobel_x = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=np.float32)
        sobel_y = sobel_x.T  # 转置得到 Y 方向核

        # 转换为 PyTorch Tensor 并调整维度 [out_channels, in_channels, H, W]
        sobel_kernel_x = torch.tensor(sobel_x).view(1, 1, 3, 3).repeat(channel, 1, 1, 1)
        sobel_kernel_y = torch.tensor(sobel_y).view(1, 1, 3, 3).repeat(channel, 1, 1, 1)

        # 定义 Sobel 卷积（使用 groups 实现 depth-wise 卷积）
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        # **使用 register_buffer 让权重成为不可训练的参数**
        self.register_buffer('sobel_kernel_x', sobel_kernel_x)
        self.register_buffer('sobel_kernel_y', sobel_kernel_y)

        # 直接赋值权重
        self.conv_x.weight.data.copy_(sobel_kernel_x)
        self.conv_y.weight.data.copy_(sobel_kernel_y)

    def forward(self, x):
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  # 计算梯度幅值，防止除 0
        # print(edge.shape)
        return edge


class SemanticEdgeFusion(nn.Module):
    def __init__(self, inc, oucs, cutoff_frequency=0.01) -> None:
        super().__init__()

        img_channel, feat_channel = inc
        self.fourier_with_attention = FourierHighPassFilterWithAttention(feat_channel, cutoff_frequency=cutoff_frequency)
        self.high_pass_filter = FourierHighPassFilter(inc, cutoff_frequency=cutoff_frequency)
        self.sc = SobelConv(feat_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1s = nn.ModuleList(Conv(feat_channel, ouc, 1) for ouc in oucs)
        self.afma = CSSG(inc, 32)
        self.current_epoch = 0
        self.targets = None
        self.register_buffer('alpha', torch.tensor(0.0))

    def forward(self, inputs):
        img, x = inputs

        x_save = x.clone()

        attentions = self.afma(img, x)
        
        patch_size = 32
        cssg_loss = None
        if self.targets is not None:
            with torch.no_grad():
                B, C, N_high, N_low = attentions.shape
                golden_attn = build_golden_attention_from_bbox_by_class(
                    self.targets,
                    img_shape=(img.shape[-2], img.shape[-1]),
                    feat_shape=(x.shape[-2], x.shape[-1]),
                    patch_size=patch_size,
                    device=img.device,
                    B=B
                )  # [B, 1, N_high, N_low]

            # 对所有通道平均（或最大）融合 → 用于监督
            pred_attn = attentions.mean(dim=1, keepdim=True)  # [B, 1, N_high, N_low]
            cssg_loss = F.mse_loss(pred_attn, golden_attn)
        # edge_feat = self.sc(x)

        B, C, H, W = x.shape
        # img_edge = F.interpolate(img_edge, size=(H, W), mode='bilinear', align_corners=False)
        # img_edge = self.edge_mapper(img_edge)
        # print(img_edge.shape)
        patch_area = patch_size * patch_size
        unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)).to(x.device)
        fold = nn.Fold(output_size=(img.shape[-2], img.shape[-1]),
                       kernel_size=(patch_size, patch_size),
                       stride=(patch_size, patch_size)).to(x.device)

        x_unfold = unfold(x)  # [B, C * patch_area, N_low]
        x_unfold = x_unfold.view(B, C, patch_area, -1)  # [B, C, patch_area, N_low]

        # att = F.softmax(attentions, dim=-1)  # [B, C, N_high, N_low]
        # mod = torch.einsum('bcij, bcjk -> bcik', att, edge_unfold.transpose(-2, -1))  # [B, C, N_high, patch_area]
        # mod = mod.permute(0, 1, 3, 2).contiguous().view(B, C * patch_area, -1)  # [B, C*patch_area, N_high]
        # edge_modulated = fold(mod)  # [B, C, H', W']

        modulated = []
        for i in range(C):
            att = F.softmax(attentions[:, i, :, :])  # [B, N_high, N_low]
            # att = attentions[:, i, :, :]  # [B, N_high, N_low]
            feat = x_unfold[:, i, :, :]  # [B, patch_area, N_low]
            # non_zeros = torch.count_nonzero(att, dim=-1).unsqueeze(-1).clamp(min=1e-5)

            # mod = torch.matmul(att / non_zeros, feat.transpose(-1, -2))  # [B, N_high, patch_area]
            mod = torch.matmul(att, feat.transpose(-1, -2))  # [B, N_high, patch_area]
            mod = mod.transpose(-1, -2).contiguous().view(B, -1, att.shape[1])  # [B, patch_area*C, N_high]

            mod = fold(mod)  # [B, 1, H', W'] → H',W' = 原图大小
            modulated.append(mod)

        x_modulated = torch.cat(modulated, dim=1)  # [B, C, H', W']

        if x_modulated.shape[-2:] != x.shape[-2:]:
            x = F.interpolate(x_modulated, size=(H, W), mode='bilinear', align_corners=False)

        x = self.alpha * x + x_save

        high_pass_filtered = self.fourier_with_attention(x)

        edge_features = self.sc(high_pass_filtered)
      
        outputs = [edge_features]

        outputs.extend(self.maxpool(outputs[-1]) for _ in self.conv_1x1s)
        outputs = outputs[1:]
        for i in range(len(self.conv_1x1s)):
            outputs[i] = self.conv_1x1s[i](outputs[i])
        self.cssg_loss = cssg_loss
        return outputs
