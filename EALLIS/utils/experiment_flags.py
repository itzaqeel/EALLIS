class ExpFlags:
    def __init__(self,
                 use_illum=True,
                 use_edge=True,
                 use_refiner=True,
                 edge_loss_weight=0.5):
        self.use_illum = use_illum
        self.use_edge = use_edge
        self.use_refiner = use_refiner
        self.edge_loss_weight = edge_loss_weight


PRESETS = {
    "baseline": ExpFlags(False, False, False, 0.0),
    "illum_only": ExpFlags(True, False, False, 0.0),
    "illum_edge": ExpFlags(True, True, False, 0.5),
    "full": ExpFlags(True, True, True, 0.5),
}
