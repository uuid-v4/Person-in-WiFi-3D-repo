_base_ = './petr_wifi_esp_like_2d_150e.py'

# Own-room ESP32 data: 3 receivers x 3 injectors (PAU0F + 2 RTL8192EU), 20 packets x 30 subcarriers
# per link -> 180 tokens of 30 values (cache csi.npy shape (N, 3, 3, 20, 30)).
cache_root = 'data/cache_9link'
data = dict(
    train=dict(cache_dir=cache_root + '/train'),
    val=dict(cache_dir=cache_root + '/test'),
    test=dict(cache_dir=cache_root + '/test'))
runner = dict(max_epochs=50)
lr_config = dict(step=[40])
evaluation = dict(interval=5, metric='pck2d')
checkpoint_config = dict(interval=5)
work_dir = 'work_dirs/petr_wifi_esp_multitx_2d'
