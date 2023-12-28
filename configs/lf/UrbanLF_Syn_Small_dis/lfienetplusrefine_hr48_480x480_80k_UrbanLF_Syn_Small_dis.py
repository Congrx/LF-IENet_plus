_base_ = [
    '../../_base_/models/lfnet_hr48.py', '../../_base_/datasets/UrbanLF_Syn_Small_dis_480x480.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
optimizer = dict(type='SGD',lr=0.01,momentum=0.9,weight_decay=0.0005)
norm_cfg = dict(type='SyncBN', requires_grad=True)
warp_channels = 512
model = dict(
    decode_head=dict(type='LFHRIENETPLUSREFINE', num_classes=14,align_corners=False, input_transform='resize_concat', sai_number=4, lf_number=25, key_channels=256, value_channels=512,warp_channels=warp_channels,disp_channels=150,dis_candidate=[-0.5/8,-0.25/8,-0/8,-0.25/8,0.5/8,0.75/8,1/8,1.25/8,1.5/8],valid_dis_range=1),
    auxiliary_head=[dict(
        type='FCNHead',
        in_channels=[48, 96, 192, 384],
        in_index=(0, 1, 2, 3),
        channels=sum([48, 96, 192, 384])//2,
        input_transform='resize_concat',
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
