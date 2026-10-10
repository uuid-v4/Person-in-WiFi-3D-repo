_base_ = './petr_wifi_cached.py'

lr_config = dict(policy='step', step=[133])
runner = dict(type='EpochBasedRunner', max_epochs=150)
work_dir = 'work_dirs/petr_wifi_cached_150e'
