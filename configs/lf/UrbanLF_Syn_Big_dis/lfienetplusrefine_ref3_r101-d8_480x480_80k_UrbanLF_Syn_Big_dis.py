_base_ = [
    '../../_base_/models/lfnet_r101-d8.py', '../../_base_/datasets/UrbanLF_Syn_Big_dis_480x480.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
optimizer = dict(type='SGD',lr=0.01,momentum=0.9,weight_decay=0.0005)
norm_cfg = dict(type='SyncBN', requires_grad=True)
warp_channels = 512
model = dict(
        decode_head=dict(type='LFIENETPLUSREFINE', num_classes=14,align_corners=False, sai_number=4, lf_number=25, key_channels=256, value_channels=512, warp_channels=warp_channels, disp_channels=150,dis_candidate=[-7/8,-5.25/8,-3.5/8,-1.75/8,0/8,1.75/8,3.5/8,5.25/8,7/8],valid_dis_range=1.0 ),
    auxiliary_head=[dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=14,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='FCNHeadDir',
        in_channels=warp_channels,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=14,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1)),
        dict(
        type='FCNHeadDir',
        in_channels=warp_channels,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=14,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1)),
        dict(
        type='FCNHeadDir',
        in_channels=warp_channels,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=14,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1))
    ])
data = dict(train=dict(sai_number=4,lf_number=25),
            val=dict(sai_number=4,lf_number=25),
            test=dict(sai_number=4,lf_number=25))
test_cfg = dict(mode='slide', crop_size=(480, 480), stride=(100, 100))
