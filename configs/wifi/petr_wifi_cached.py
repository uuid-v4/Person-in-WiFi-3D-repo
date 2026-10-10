_base_ = './petr_wifi.py'

cache_root = 'data/wiception3d_cache'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=16,
    train=dict(
        _delete_=True,
        type='opera.WifiPoseCachedDataset',
        cache_dir=cache_root + '/train',
        pipeline={{_base_.train_pipeline}}),
    val=dict(
        _delete_=True,
        type='opera.WifiPoseCachedDataset',
        cache_dir=cache_root + '/test',
        pipeline={{_base_.test_pipeline}}),
    test=dict(
        _delete_=True,
        type='opera.WifiPoseCachedDataset',
        cache_dir=cache_root + '/test',
        pipeline={{_base_.test_pipeline}}))
evaluation = dict(interval=10, metric='mpjpe')
checkpoint_config = dict(interval=10, max_keep_ckpts=5)
work_dir = 'work_dirs/petr_wifi_cached'
gpu_ids = range(0, 1)
