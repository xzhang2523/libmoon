

class CoreMGDA:
    def __init__(self):
        pass








class CoreGrad:
    def __init__(self):
        pass

    def get_gw(self, G, losses, pref):
        pass

class CoreEPO(CoreGrad):
    def __init__(self):
        pass

    def get_gw(self, G, losses, pref):
        pass


class CoreAgg(CoreGrad):
    def __init__(self, agg_mtd='ls'):
        self.agg_mtd = agg_mtd

    def get_alpha(self, losses, pref):
        if self.agg_mtd == 'ls':
            return pref
        else:
            assert False