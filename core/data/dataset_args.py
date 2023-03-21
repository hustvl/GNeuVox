from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394','female-4-casual','female-3-casual','male-4-casual','male-3-casual']

    for sub in subjects:
        dataset_attrs.update({
            f"{sub}_train": {
                "dataset_path": f"processed_data/{sub}",
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            f"{sub}_test": {
                "dataset_path": f"processed_data/{sub}", 
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'mocap'
            },
        })

    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
