import mxnet as mx
from mxnet.gluon import nn, Block

class CosineSoftmaxClassifier(Block):
    def __init__(self, dim, num_classes):
        super(CosineSoftmaxClassifier, self).__init__()
        with self.name_scope():
            self.kappa = self.params.get('kappa', shape=(1,))
            self.weight = self.params.get(
                'weight', init=mx.init.Xavier(magnitude=2.24),
                shape=(dim, num_classes))

    def forward(self, logits):
        with logits.context:
            #normalize to unit sphere
            norms = mx.nd.norm(self.weight.data(), axis=0)
            norm_weight = self.weight.data() / norms
            #apply weights
            linear = mx.nd.dot(logits, norm_weight)
            #scale by kappa
            weighted_linear = self.kappa.data() * linear
            #get probs via softmax
            probs = mx.nd.softmax(weighted_linear,axis=1)
            return probs

ctx = mx.cpu()
clf = CosineSoftmaxClassifier(4, 10)
clf.collect_params().initialize(ctx=ctx)

a = mx.nd.array([[4, 3, 2, 1], [5, 4, 3, 2]])
clf(a)
