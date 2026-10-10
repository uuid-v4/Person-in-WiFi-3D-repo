_base_ = './petr_wifi_esp_like_150e.py'

cache_root = 'data/wiception3d_cache_esp_2d'
data = dict(
    train=dict(cache_dir=cache_root + '/train'),
    val=dict(cache_dir=cache_root + '/test'),
    test=dict(cache_dir=cache_root + '/test'))
model = dict(bbox_head=dict(
    coord_dim=2,
    transformer=dict(coord_dim=2)))
evaluation = dict(interval=10, metric='pck2d')
work_dir = 'work_dirs/petr_wifi_esp_like_2d_150e'
