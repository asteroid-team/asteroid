import model
import torch as th
import yaml
from ipdb import set_trace

set_trace()

#m = model.ChimeraPP('lstm', 2, 401)
#m(th.rand(10, 401, 20))

with open('conf.yml') as f:
    def_conf = yaml.safe_load(f)
model.make_model_and_optimizer(def_conf)

