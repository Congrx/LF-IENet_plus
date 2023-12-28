import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from mmseg.ops import resize
from mmcv.cnn import ConvModule
from .decode_head import BaseDecodeHead
from ..builder import HEADS
from ..utils import SequenceConv
import math
from einops import rearrange

class CrossFrameAttention(nn.Module):
    def __init__(self, valid_dis_range, matmul_norm=False):
        super(CrossFrameAttention, self).__init__()
        self.matmul_norm = matmul_norm
        self.radius = valid_dis_range / 8     
        self.weight = 0.2       
        self.valid_large_attn = 1
        
    def forward(self, memory_keys, memory_values, query_query, disparity, sequence_index):
        sai_number, batch_size, key_channels, height, width = memory_keys.shape
        _, _, value_channels, _, _ = memory_values.shape
        assert query_query.shape[1] == key_channels
        memory_keys = memory_keys.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_keys = memory_keys.view(batch_size, key_channels, sai_number * height * width)  # BxCxT*H*W
        query_query = query_query.view(batch_size, key_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCk
        key_attention = torch.bmm(query_query, memory_keys)  # BxH*WxT*H*W
        
        weight = self.weight / sai_number / height / width
        large_dis = []
        for i in range(sai_number):   
            atten_mask = torch.zeros([batch_size,1,height,width]).cuda()   
            distance = torch.sqrt((sequence_index[:,i,1]-5)**2 + (sequence_index[:,i,0]-5)**2)  
            total_disparity = distance.reshape(batch_size,1,1,1) * disparity     
            # shield small disparity perception
            atten_mask[total_disparity > self.radius] = 1     
            atten_mask[total_disparity < -1 * self.radius] = 1    
            large_dis.append(atten_mask)
        
        large_dis = torch.stack(large_dis,2)    
        large_dis = large_dis.view(batch_size,1,sai_number*height*width)  
        large_dis = large_dis.float().masked_fill(large_dis == 0, float('-inf')). masked_fill(large_dis == 1, float(0.0))               
        
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = key_attention + large_dis
        key_attention = F.softmax(key_attention, dim=-1)  
        
        memory_values = memory_values.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_values = memory_values.view(batch_size, value_channels, sai_number * height * width)
        memory_values = memory_values.permute(0, 2, 1).contiguous()  # BxT*H*WxC
        memory = torch.bmm(key_attention, memory_values)  # BxH*WxC
        memory = memory.permute(0, 2, 1).contiguous()  # BxCxH*W
        memory = memory.view(batch_size, value_channels, height, width)  # BxCxHxW
        return memory

class SelfFrameAttention(nn.Module):
    def __init__(self,
                 matmul_norm=False):
        super(SelfFrameAttention, self).__init__()
        self.matmul_norm = matmul_norm

    def forward(self, query_query, query_key, query_value):
        batch_size, key_channels, height, width = query_query.shape
        _, value_channels, _, _ = query_value.shape
        query_query = query_query.view(batch_size, key_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCk
        query_key = query_key.view(batch_size, key_channels, height * width)  # BxCkxH*W
        key_attention = torch.bmm(query_query, query_key)  # BxH*WxH*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=-1)  # BxH*WxH*W
        
        query_value = query_value.view(batch_size, value_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCv
        memory = torch.bmm(key_attention, query_value)  # BxH*WxCv
        memory = memory.permute(0, 2, 1).contiguous()  # BxCvxH*W
        memory = memory.view(batch_size, value_channels, height, width)  # BxCxHxW
        return memory


class SELayer(nn.Module):
    def __init__(self, out_ch, g=16):
        super(SELayer, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // g, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // g, out_ch, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, feature):
        x = F.adaptive_avg_pool2d(feature, (1, 1))
        attn = self.attn(x)
        feature = feature * attn
        return feature

class feature_extraction(nn.Module):
    def __init__(self, input_dim, device=None):
        super(feature_extraction, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.layers = list()
        input_dim = 4
        numblock = [2, 8, 2, 2]
        output_dim = [4, 8, 16, 16]
        for i in range(0, 4):
            if i >= 3:
                stride = 1
            else:       
                stride = 2
            temp = self._make_layer(input_dim, output_dim[i], numblock[i], stride)
            self.layers.append(temp)
            input_dim = output_dim[i]
        self.layers = nn.Sequential(*self.layers)
        # SPP Module
        self.branchs = list()
        output_dim = [4, 4, 4, 4]
        size = [2, 4, 8, 16]
        for i in range(0, 4):
            temp = nn.Sequential(
                nn.AvgPool2d((size[i], size[i]), (size[i], size[i])),
                nn.Conv2d(input_dim, output_dim[i], kernel_size=1, stride=1, dilation=1),
                nn.BatchNorm2d(output_dim[i]),
                nn.ReLU(),
            )  
            self.branchs.append(temp)
        self.branchs = nn.Sequential(*self.branchs)
        input_dim = np.array(output_dim).sum() + 16
        self.last = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1,dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, bias=False)
        )

    def _make_layer(self, input_dim, out_channels, blocks, stride):
        layers = list()
        layers.append(BasicBlock(input_dim, out_channels, stride))
        for i in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        layers = nn.Sequential(*layers)
        return layers

    def forward(self, x):
        # x : B*MM, C, H, W
        x = self.conv1(x)
        layers_out = [x]
        for i in range(len(self.layers)):
            layers_out.append(self.layers[i](layers_out[-1]))

        layer4_size = layers_out[-1].shape  # B,C,H,W
        branchs_out = []
        for i in range(len(self.branchs)):
            temp = self.branchs[i](layers_out[-1])
            temp = nn.UpsamplingBilinear2d(size=(int(layer4_size[-2]), int(layer4_size[-1])))(temp)
            branchs_out.append(temp)

        cat_f = [layers_out[4]] + branchs_out
        feature = torch.cat([i for i in cat_f], dim=1)

        out = self.last(feature)
        # x : B, C,H, W
        return out

class BasicBlock(nn.Module):

    def __init__(self, input_dim, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_channels, kernel_size=3, stride=stride, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or out_channels != 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_dim, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.downsample(x)
        out = x1 + x2
        return out

class Basic(nn.Module):
    def __init__(self, views, disp_channel, device=None):
        super(Basic, self).__init__()
        self.views = views
        self.disp_channel = disp_channel
        self.conv1 = nn.Sequential(
            nn.Conv3d(4 * self.views, self.disp_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(self.disp_channel),
            nn.ReLU(),
            nn.Conv3d(self.disp_channel, self.disp_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(self.disp_channel),
            nn.ReLU()
        )
        self.basics = list()
        for i in range(0, 2):
            temp = nn.Sequential(
                nn.Conv3d(self.disp_channel, self.disp_channel, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm3d(self.disp_channel),
                nn.ReLU(),
                nn.Conv3d(self.disp_channel, self.disp_channel, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm3d(self.disp_channel),
            )
            self.basics.append(temp)
        self.basics = nn.Sequential(*self.basics)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(0, len(self.basics)):
            x = (x + self.basics[i](x))/2
        return x

@HEADS.register_module()
class LFIENETPLUSREFINE(BaseDecodeHead):
    def __init__(self, sai_number, lf_number, key_channels, value_channels, warp_channels, disp_channels, dis_candidate, valid_dis_range, **kwargs):
        super(LFIENETPLUSREFINE, self).__init__(**kwargs)
        self.sai_number = sai_number
        self.valid_dis_range = valid_dis_range
        self.lf_number = lf_number
        self.warp_channel = warp_channels
        self.disp_channel = disp_channels
        self.dis_candidate = dis_candidate
        self.reference_encoding = nn.Sequential(
            SequenceConv(self.in_channels, value_channels, 1, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(value_channels, value_channels, 3, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        
        self.reference_key_conv = nn.Sequential(
            SequenceConv(value_channels, key_channels, 1, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(key_channels, key_channels, 3, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        self.reference_value_conv = nn.Sequential(
            SequenceConv(value_channels, value_channels, 1, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(value_channels, value_channels, 3, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        
        self.warp_feature_conv = nn.Sequential(
            SequenceConv(self.in_channels, self.warp_channel, 1, sai_number,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(self.warp_channel, self.warp_channel, 3, sai_number,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        
        self.center_encoding = nn.Sequential(
            ConvModule(
                self.in_channels,
                value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                value_channels,
                value_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        self.center_query_conv = nn.Sequential(
            ConvModule(
                value_channels,
                key_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                key_channels,
                key_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        self.center_key_conv = nn.Sequential(
            ConvModule(
                value_channels,
                key_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                key_channels,
                key_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        
        self.center_value_conv = nn.Sequential(
            ConvModule(
                value_channels,
                value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                value_channels,
                value_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        self.cross_attention = CrossFrameAttention(self.valid_dis_range, matmul_norm=False)
        self.self_attention = SelfFrameAttention(matmul_norm=False)
        
        self.channel_attention = SELayer(self.warp_channel*(sai_number-1))

        self.fuse_warp_conv = nn.Sequential(
            ConvModule(
                self.warp_channel*(sai_number-1),
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        self.bottleneck_1 = ConvModule(
            value_channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        self.bottleneck_2 = ConvModule(
            self.channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        # semantic-aware disparity estimator
        self.feature_extraction_layer = feature_extraction(1)
        self.channel_attention_dis = nn.Sequential(
            nn.Conv3d(4 * self.lf_number , 170, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(170, self.lf_number, kernel_size=1),
            nn.Sigmoid(),
        )
        self.basics = Basic(views=self.lf_number,disp_channel=self.disp_channel)
        
        self.dis_cls = nn.Sequential(
            nn.Conv3d(self.disp_channel, self.disp_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(self.disp_channel),
            nn.ReLU(),
            nn.Conv3d(self.disp_channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )
        
        self.dis_query_conv = nn.Sequential(
            ConvModule(
                self.disp_channel,
                self.disp_channel,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                self.disp_channel,
                self.disp_channel,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        self.dis_key_conv = nn.Sequential(
            ConvModule(
                self.disp_channel,
                self.disp_channel,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                self.disp_channel,
                self.disp_channel,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        
        self.dis_value_conv = nn.Sequential(
            ConvModule(
                self.disp_channel,
                self.disp_channel,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                self.disp_channel,
                self.disp_channel,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        self.dis_encoding = nn.Sequential(
            ConvModule(
                self.disp_channel,
                self.disp_channel,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                self.disp_channel,
                self.disp_channel,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        
        
    def forward(self, inputs, seg_logits, sequence_imgs, sequence_warp_imgs, lf_imgs, sequence_index): 
        center_img = self._transform_inputs(inputs)
        sequence_imgs = [self._transform_inputs(inputs).unsqueeze(0) for inputs in sequence_imgs]  # T, BxCxHxW
        sequence_imgs = torch.cat(sequence_imgs, dim=0)  # TxBxCxHxW
        sai_number, batch_size, channels, height, width = sequence_imgs.shape
        assert sai_number == self.sai_number - 1
        
        ''' get disparity prior from semantic-aware disparity estimator '''
        # step1: Feature Extraction
        B, views, C, H, W = lf_imgs.shape  
        lf_imgs = lf_imgs.reshape(B * views, C, H, W)  
        lf_feature = self.feature_extraction_layer(lf_imgs)  
        _, _, H, W = center_img.shape
        lf_feature = lf_feature.reshape(B, views, -1, H, W).permute(1, 0, 2, 3, 4)  
        view_list = list()          
        for i in range(0, views):
            view_list.append(lf_feature[i, :])
        
        # step2: Semantic-aware Cost Construction
        disparity_costs = list()    
        disparity_number = len(self.dis_candidate)        
        disparity_sub = []       
        for d in self.dis_candidate:
            if d == 0:
                tmp_list = list()
                for i in range(len(view_list)):
                    tmp_list.append(view_list[i])
            else:
                tmp_list = list()
                if views == 25:
                    lf_index = [20,21,22,23,24,29,30,31,32,33,38,39,40,41,42,47,48,49,50,51,56,57,58,59,60]
                elif views == 17:
                    lf_index = [20,22,24,30,31,32,38,39,40,41,42,48,49,50,56,58,60] 
                for i in range(len(view_list)):
                    (v, u) = divmod(lf_index[i], 9) 
                    rate = [2*d * (v - 4) / H, 2*d * (u - 4) / W]
                    theta = torch.tensor([[1, 0, rate[0]], [0, 1, rate[1]]], dtype=float).cuda()
                    grid = F.affine_grid(theta.unsqueeze(0).repeat(B, 1, 1), view_list[i].size()).type_as(view_list[i])
                    temp = F.grid_sample(view_list[i], grid,mode='bilinear',padding_mode='border')
                    tmp_list.append(temp)
            cost = torch.cat([i for i in tmp_list], dim=1)
            disparity_costs.append(cost)
        cost_volume = torch.cat([i.unsqueeze(dim=1) for i in disparity_costs], dim=1)  
        cost_volume = cost_volume.reshape(B, disparity_number, 4*views, H, W).permute(0, 2, 3, 4, 1)        # get initial cost volume
        
        x = torch.nn.functional.adaptive_avg_pool3d(cost_volume, (1, 1, 1))
        x = self.channel_attention_dis(x)  
        attention = x.repeat(1, 4, 1, 1, 1)  
        cv = attention * cost_volume        
        dis_feature = self.basics(cv)           # get view-weighted cost volume
        
        batch_size, num_classes, _, _ = seg_logits.size()
        probs = seg_logits.view(batch_size, num_classes, -1)      
        probs = F.softmax(probs, dim=2)   
        probs = probs.unsqueeze(1).expand(-1,disparity_number,-1,-1)      
        probs = probs.reshape(batch_size*disparity_number,num_classes,H*W)              # (batch*level) × num_classes × HW
        dis_feature_ori = rearrange(dis_feature,'b c h w level -> (b level) c h w')     
        dis_feature = dis_feature_ori.view(batch_size*disparity_number,self.disp_channel,-1).permute(0,2,1)   # (batch×level) × HW × self.disp_channel
        dis_context = torch.matmul(probs, dis_feature)              
        dis_context = dis_context.permute(0, 2, 1).contiguous().unsqueeze(3)        # (batch*level) × self.disp_channel × num_classes × 1   ------>    get Object Region Representation
        
        dis_feature_query = self.dis_query_conv(dis_feature_ori)  
        dis_context_key = self.dis_key_conv(dis_context)  
        dis_context_value = self.dis_value_conv(dis_context)  
        batch_size_total, dis_channels , height, width = dis_feature_query.shape
        dis_feature_query = dis_feature_query.view(batch_size_total, dis_channels, -1).permute(0, 2, 1).contiguous()  # (batch*level) × HW × self.disp_channel
        dis_context_key = dis_context_key.view(batch_size_total, dis_channels, -1)  # (batch*level) × self.disp_channel × num_classes
        dis_key_attention = torch.bmm(dis_feature_query, dis_context_key)  # (batch*level) × HW × num_classes
        dis_key_attention = F.softmax(dis_key_attention, dim=-1)  #(batch*level) × HW × num_classes
        dis_context_value = dis_context_value.view(batch_size_total, dis_channels, -1).permute(0, 2, 1).contiguous()  # (batch*level) × num_classes × self.disp_channel
        dis_memory = torch.bmm(dis_key_attention, dis_context_value)  # (batch*level) × HW × self.disp_channel
        dis_memory = dis_memory.permute(0, 2, 1).contiguous()  # (batch*level) × self.disp_channel × HW
        dis_memory = dis_memory.view(batch_size_total, dis_channels, height, width)  # (batch*level) × self.disp_channel × H × W
        dis_feature = self.dis_encoding(dis_memory) + dis_feature_ori
        dis_feature = rearrange(dis_feature,'(b level) c h w -> b c h w level',b=batch_size,level=disparity_number)     # batch × self.disp_channel × H × W × level     ------>    get semantic-aware cost volume
        
        # step3: Cost Aggregation & Regression
        dis_logits = self.dis_cls(dis_feature)      # get pixel-wise cost volume
        dis_logits = dis_logits.squeeze(dim=1)
        dis_logits = nn.functional.softmax(dis_logits, dim=-1)
        disparity_values = np.array(self.dis_candidate)
        disparity_values = torch.from_numpy(disparity_values).cuda()    
        disparity_values = disparity_values.reshape(1, 1, 1, disparity_number)
        x = disparity_values.repeat(B, H, W, 1)
        out = (x * dis_logits).sum(dim=-1)
        disparity_8 = out.unsqueeze(1)  # batch × 1 × H × W
        _, _, height, width = disparity_8.shape
        
        if len(sequence_index.size()) == 4:
            sequence_index = sequence_index.squeeze(1)  # batch * 1 * sai_number-1 * 2 -> batch * sai_number-1 * 2

        reference_encoding_feature = self.reference_encoding(sequence_imgs)
        reference_key = self.reference_key_conv(reference_encoding_feature)
        reference_value = self.reference_value_conv(reference_encoding_feature)
        
        center_encoding_feature = self.center_encoding(center_img)
        center_query = self.center_query_conv(center_encoding_feature)  # BxCxHxW
        center_key = self.center_key_conv(center_encoding_feature)  # BxCxHxW
        center_value = self.center_value_conv(center_encoding_feature)  # BxCxHxW
        
        
        ''' Implicit Feature Integration '''
        cross_feature = self.cross_attention(reference_key, reference_value, center_query, disparity_8, sequence_index) + center_encoding_feature
        self_feature = self.self_attention(center_query, center_key, center_value) + center_encoding_feature
        output = torch.cat([self_feature, cross_feature], dim=1)
        
        ''' Explicit Feature Propagation '''
        disparity_8_reshape = disparity_8.permute(0, 2, 3, 1).contiguous()    
        x = np.array([i for i in range(0, height)]).reshape(1, height, 1, 1).repeat(repeats=width, axis=2)
        y = np.array([i for i in range(0, width)]).reshape(1, 1, width, 1).repeat(repeats=height, axis=1)
        xy_position = torch.from_numpy(np.concatenate([x, y], axis=-1))     
        coords_x = torch.linspace(-1, 1, width).to(torch.float32)      
        coords_y = torch.linspace(-1, 1, height).to(torch.float32)       
        coords_x = coords_x.repeat(height, 1).reshape(height, width, 1)          
        coords_y = coords_y.repeat(width, 1).permute(1, 0).reshape(height, width, 1)
        coords = torch.cat([coords_x, coords_y], dim=2)             
        coords = coords.reshape(1, height, width, 2)       
        coords = coords[0, xy_position[:, :, :, 0].reshape(-1).to(torch.int64),xy_position[:, :, :, 1].reshape(-1).to(torch.int64), :]   
        coords = coords.reshape(-1, height, width, 2).cuda()                    
        coords = coords.repeat(batch_size,1,1,1)
        
        warp_features = []
        dst_u = 5   # center_view_coord
        dst_v = 5   # center_view_coord
        sequence_imgs = self.warp_feature_conv(torch.cat([sequence_imgs,center_img.unsqueeze(0)],0))
        center_img = sequence_imgs[sai_number]
        
        current_center = sequence_warp_imgs[:,sai_number,:,:,:]     
        current_center = resize(
                current_center,
                size=sequence_imgs.shape[3:],
                mode='bilinear',
                warning=False)              # center view grey image  for image-level warping
        # get final aligned feature from reference view to central view via image-level warping and feature-level warping
        for i in range(sai_number):                     
            current_feature = sequence_imgs[i]           # center view feature
            current_img = sequence_warp_imgs[:,i,:,:,:]     # reference view grey image
            current_img = resize(
                current_img,
                size=sequence_imgs.shape[3:],
                mode='bilinear',
                warning=False)  
            offsetx = 2 * (dst_u - sequence_index[:,i,1]).reshape(batch_size,1,1,1) * disparity_8_reshape[:, :, :, :]      
            offsety = 2 * (dst_v - sequence_index[:,i,0]).reshape(batch_size,1,1,1) * disparity_8_reshape[:, :, :, :]     
            coords_x = (coords[:, :, :, 0:1] * width + offsetx) / width     
            coords_y = (coords[:, :, :, 1:2] * height + offsety) / height    
            coords_uv = torch.cat([coords_x, coords_y], dim=-1).to(torch.float32)    
            warp_img = F.grid_sample(current_img, coords_uv[:, :, :, :],mode='bilinear',padding_mode='border')     #  warp reference view grey image
            curr_mask = torch.abs(warp_img-current_center)
            weight = (1-curr_mask) ** 2                             # generate propagation mask via image-level warping
            temp = F.grid_sample(current_feature, coords_uv[:, :, :, :],mode='bilinear',padding_mode='border')     #  warp reference view feature
            warp_features.append(center_img*(1-weight)+temp*weight)     # get final aligned feature
        warp_feature = torch.cat(warp_features, 1) 
        warp_feature_attention = self.channel_attention(warp_feature)
        warp_feature_final = self.fuse_warp_conv(warp_feature_attention)

        ''' Feature Fusion & Decoding '''
        output = self.bottleneck_1(output)      
        output = self.bottleneck_2(torch.cat([warp_feature_final, output], dim=1))     
        output = self.cls_seg(output)          
        return output,warp_features   

