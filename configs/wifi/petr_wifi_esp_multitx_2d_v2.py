_base_ = './petr_wifi_esp_multitx_2d.py'

# v2 (csi_demo/train/IMPROVEMENTS.md): cache built with
#   build_2d_cache.py ... --tx-ids 1,2,3 --phase --packet-norm --label-smooth 5 --out data/cache_9link_v2
# -> 30 amplitude + 30 sanitized-phase values per packet; per-joint visibility masks the loss;
# link dropout + noise augmentation; stronger weight decay; cosine LR; keep the best checkpoint by pck@0.1.
cache_root = 'data/cache_9link_v2'
model = dict(csi_in_channels=60)
data = dict(
    train=dict(cache_dir=cache_root + '/train', use_vis=True, link_dropout=0.1, noise_std=0.02),
    val=dict(cache_dir=cache_root + '/test'),
    test=dict(cache_dir=cache_root + '/test'))
optimizer = dict(weight_decay=0.05)
lr_config = dict(_delete_=True, policy='CosineAnnealing', warmup='linear', warmup_iters=500,
                 warmup_ratio=0.1, min_lr_ratio=0.01)
runner = dict(max_epochs=40)
evaluation = dict(interval=5, metric='pck2d', save_best='pck@0.1', rule='greater')
checkpoint_config = dict(interval=5)
work_dir = 'work_dirs/petr_wifi_esp_multitx_2d_v2'
