class AMCosineSoftmaxClf(Block):
	def __init__(self, dim, num_classes, kappa=10, margin=2):
		supe(AMCosineSoftmaxClf, self).__init__()
		self.kappa = kappa
		self.margin = margin

		with self.name_scope():
			self.weight = self.params.get(
				'weight', init=mx.init.Xavier(magnitude=2.0),
				shape=(dim, num_classes))

	def forward(self, logits, labels):
		with logits.context:
			# normalize to unit sphere
			norms = mx.nd.norm(self.weight.data(), axis=0)
			norm_weight = self.weight.data() / norms

			#apply weights
			linear = mx.nd.dot(logits, norm_weight)
			linear = mx.nd.clip(linear, -1, 1)

			#compute adjusted log softmax
	    	numerator = self.kappa * (mx.nd.linalg.extractdiag(mx.nd.transpose(linear, (1, 0))[labels]) - self.margin)
		    excl_dims = [mx.nd.concat(linear[i, :int(y.asscalar())], linear[i, int(y.asscalar())+1:], dim=0) for i, y in enumerate(labels)]
		    excl = mx.nd.stack(*excl_dims, axis=0)
		    denominator = mx.nd.exp(numerator) + mx.nd.sum(mx.nd.exp(self.kappa * excl), axis=1)
		    logits = numerator - mx.nd.log(denominator)
	        return logits