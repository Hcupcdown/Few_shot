from .RadarMossFormer.radar_mossformer import RadarMossFormer
from .RadarMossFormer.radar_net import RadarNet


def bulid_model(args):
    model = RadarMossFormer()
    return model