_base_ = './petr_wifi_esp_multitx_2d_v2.py'

# Fine-tune v2 (epoch 25) on yesterday's crowded sessions + today's quiet session, 1.0 s windows as used live:
#   build_2d_cache.py ... s1 s2 s4 (+ s5 test) --phase --packet-norm --label-smooth 5 --max-age 1.0 --out data/cache_9link_v2ft
cache_root = 'data/cache_9link_v2ft'
data = dict(
    train=dict(cache_dir=cache_root + '/train'),
    val=dict(cache_dir=cache_root + '/test'),
    test=dict(cache_dir=cache_root + '/test'))
load_from = 'work_dirs/petr_wifi_esp_multitx_2d_v2/epoch_25.pth'
lr_config = dict(warmup_iters=200)
runner = dict(max_epochs=8)
evaluation = dict(interval=2, metric='pck2d', save_best='pck@0.1', rule='greater')
checkpoint_config = dict(interval=2)
work_dir = 'work_dirs/petr_wifi_esp_multitx_2d_v2ft'
