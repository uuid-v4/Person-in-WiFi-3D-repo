_base_ = './petr_wifi_cached_150e.py'

cache_root = 'data/wiception3d_cache_esp'
num_query = 50
model = dict(
    csi_in_channels=30,
    bbox_head=dict(num_query=num_query, transformer=dict(two_stage_num_proposals=num_query)),
    test_cfg=dict(max_per_img=num_query))
data = dict(
    train=dict(cache_dir=cache_root + '/train'),
    val=dict(cache_dir=cache_root + '/test'),
    test=dict(cache_dir=cache_root + '/test'))
work_dir = 'work_dirs/petr_wifi_esp_like_150e'
