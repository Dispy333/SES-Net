class FourierHighPassFilterWithAttention(nn.Module):
    def __init__(self, channel, cutoff_frequency=0.01,
                 use_attention=True, attention_type="cross",
                 attention_channels=64) -> None:
        super().__init__()

        self.cutoff_frequency = cutoff_frequency
        self.channel = channel
        self.use_attention = use_attention
        self.attention_type = attention_type

        # Attention mechanism relevant parameters
        if self.use_attention:
            self.attention_channels = attention_channels

            # 通道注意力（用于加权不同通道的重要性）
            # self.channel_attention = nn.Sequential(
            #     nn.AdaptiveAvgPool2d(1),
            #     nn.Conv2d(channel, channel // 4, kernel_size=1),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(channel // 4, channel, kernel_size=1),
            #     nn.Sigmoid()
            # )
            self.cbam = CBAM(channel, reduction=4)
            self.channel_Att = CBAM_Channel_Att(channel, reduction=4)
            self.highAtt = HighFreqAttention(channel, reduction=4)
            self.lowAtt = LowFreqAttention(channel, reduction=4, window_size=11)

            # 交叉/自注意力机制
            if attention_type == "cross":
                # 交叉注意力模块（低频到高频，高频到低频）
                # self.cross_attn_low2high = AttentionBlock(channel, attention_channels)
                # self.cross_attn_high2low = AttentionBlock(channel, attention_channels)
                self.cross_attn_low2high = CrossWindowAttention(channel, window_sizes=[(1, 7), (7, 1), (7, 7)], mode="high")
                self.cross_attn_high2low = CrossWindowAttention(channel, window_sizes=[(11, 11)], mode="low")
            elif attention_type == "self":
                # 自注意力模块
                self.self_attn = AttentionBlock(2 * channel, attention_channels)

            # 特征融合模块
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(2 * channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        device = x.device

        # 原始傅里叶变换流程
        x_freq = torch.fft.fft2(x)
        x_freq_shifted = torch.fft.fftshift(x_freq)

        h, w = x.shape[2], x.shape[3]
        cy, cx = h // 2, w // 2
        y, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w))
        y = y.to(device)
        x_grid = x_grid.to(device)

        freq_map = torch.sqrt((x_grid - cx).float() ** 2 + (y - cy).float() ** 2)
        cutoff = self.cutoff_frequency * max(h, w)
        high_pass_filter = (freq_map > cutoff).float().to(device)
        # cutoff = self.cutoff_frequency.to(device) * max(h, w)
        # high_pass_filter = torch.sigmoid((freq_map - cutoff) / self.temperature)
        low_pass_filter = 1 - high_pass_filter 

        # 同时获取高频和低频信息
        x_freq_high = x_freq_shifted * high_pass_filter
        x_freq_low = x_freq_shifted * low_pass_filter

        # 逆变换得到空间域的高、低频分量
        x_high = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_freq_high)))
        x_low = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_freq_low)))

        # 应用注意力机制增强特征
        if self.use_attention:
            # 通道注意力加权
            x_high = self.channel_Att(x_high)
            x_low = self.channel_Att(x_low)

            # 不同类型的注意力交互
            if self.attention_type == "cross":
                # 交叉注意力：低频增强高频，高频增强低频
                enhanced_high = self.cross_attn_low2high(x_low, x_high)
                enhanced_low = self.cross_attn_high2low(x_high, x_low)
            elif self.attention_type == "self":
                # 自注意力：融合后的特征自我关注
                combined = torch.cat([x_low, x_high], dim=1)
                enhanced = self.self_attn(combined, combined)
                enhanced_low, enhanced_high = torch.split(enhanced, self.channel, dim=1)

            # 特征融合与增强
            combined_features = torch.cat([enhanced_low, enhanced_high], dim=1)
            x_filtered = self.fusion_conv(combined_features)

            # 残差连接保持原始信息
            x_filtered = x_filtered + x
        else:
            # 无注意力时仅返回高频信息
            x_filtered = x_high

        return x_filtered


class CBAM_Channel_Att(nn.Module):
    def __init__(self, channels, reduction=4, kernel_size=1):
        super(CBAM_Channel_Att, self).__init__()

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attn(x)
        x = x * ca
        return x


class CrossWindowAttention(nn.Module):
    def __init__(self, dim, window_sizes=[(1, 7), (7, 1), (7, 7)], mode="high"):
        """
        Cross-window attention 模块
        Args:
            dim (int): 输入通道数
            window_sizes (list of tuple): 窗口大小集合
            mode (str): 'high' 高频分支（多窗口融合），'low' 低频分支（单大窗口）
        """
        super().__init__()
        self.dim = dim
        self.window_sizes = window_sizes
        self.mode = mode

        # QKV projection
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # 高频分支需要额外融合
        if self.mode == "high":
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(len(window_sizes) * dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )

    def forward(self, x_src, x_tgt):
        """
        Args:
            x_src: 源特征 (B, C, H, W)，作为 query
            x_tgt: 目标特征 (B, C, H, W)，作为 key, value
        """
        B, C, H, W = x_src.shape

        outputs = []
        for (Wh, Ww) in self.window_sizes if self.mode == "high" else [self.window_sizes[0]]:
            # padding 保证整除
            pad_h = (Wh - H % Wh) % Wh
            pad_w = (Ww - W % Ww) % Ww
            src = F.pad(x_src, (0, pad_w, 0, pad_h))
            tgt = F.pad(x_tgt, (0, pad_w, 0, pad_h))
            Hp, Wp = src.shape[2], src.shape[3]

            # 分窗口
            def window_partition(x, Wh, Ww):
                B, C, H, W = x.shape
                x = x.view(B, C, H // Wh, Wh, W // Ww, Ww)
                x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
                return x.view(-1, Wh * Ww, C)  # (Bn, N, C)

            def window_reverse(windows, Wh, Ww, H, W):
                Bn, Np, C = windows.shape
                B = Bn // ((H // Wh) * (W // Ww))
                x = windows.view(B, H // Wh, W // Ww, Wh, Ww, C)
                x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
                return x.view(B, C, H, W)

            src_windows = window_partition(src, Wh, Ww)
            tgt_windows = window_partition(tgt, Wh, Ww)

            # QKV
            Q = self.q_proj(src_windows)
            K = self.k_proj(tgt_windows)
            V = self.v_proj(tgt_windows)

            attn = (Q @ K.transpose(-2, -1)) / (C ** 0.5)
            attn = F.softmax(attn, dim=-1)
            out = attn @ V
            out = self.out_proj(out)

            # 拼回去
            out = window_reverse(out, Wh, Ww, Hp, Wp)
            out = out[:, :, :H, :W]  # 去掉 padding
            outputs.append(out)

        # 高频 → 多窗口融合
        if self.mode == "high":
            out = torch.cat(outputs, dim=1)  # (B, C*k, H, W)
            out = self.fuse_conv(out)
        else:
            out = outputs[0]

        return out


class AttentionBlock(nn.Module):
    """通用注意力机制模块"""

    def __init__(self, in_channels, inner_channels):
        super().__init__()

        # 查询、键、值的投影层
        self.query_conv = nn.Conv2d(in_channels, inner_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, inner_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, inner_channels, kernel_size=1)

        # 输出投影
        self.output_conv = nn.Conv2d(inner_channels, in_channels, kernel_size=1)

        # 归一化和激活
        self.norm = nn.BatchNorm2d(inner_channels)
        self.activation = nn.ReLU(inplace=True)

        # 缩放因子
        self.scale = torch.sqrt(torch.tensor(inner_channels).float())

    def forward(self, x, context=None):
        context = context if context is not None else x

        # 投影到查询、键、值空间
        query = self.query_conv(x)
        key = self.key_conv(context)
        value = self.value_conv(context)

        # 注意力分数计算
        attn_scores = torch.einsum('bchw,bcHW->bhwHW', query, key) / self.scale
        attn_weights = F.softmax(attn_scores.view(*attn_scores.shape[:3], -1), dim=-1)
        attn_weights = attn_weights.view(attn_scores.shape)

        # 应用注意力权重
        output = torch.einsum('bhwHW,bcHW->bchw', attn_weights, value)

        # 通过MLP和残差连接
        output = self.norm(output)
        output = self.activation(output)
        output = self.output_conv(output)

        return output + x  # 残差连接
