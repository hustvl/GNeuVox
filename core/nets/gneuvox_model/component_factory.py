import imp

def load_positional_embedder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).get_embedder

def load_canonical_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).CanonicalMLP

def load_voxel(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).voxel_model

def load_mweight_vol_decoder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).MotionWeightVolumeDecoder

def load_pose_refine(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).BodyPoseRefiner

