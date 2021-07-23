import tensorflow as tf

class SVGD:
    def __init__(self, kernel, targetDistribution, learningRate):
        self.kernel = kernel
        self.targetDistribution = targetDistribution
        self.optimizer = tf.keras.optimizers.Adam(learningRate)

    def update(self, x, nIterations):
        for _ in range(nIterations):
            kernelMatrix, kernelGrad = self.computeKernel(x)
            logprobGradient = self.logprobGradient(x)
            
            completeGrad = -(kernelMatrix @ logprobGradient + kernelGrad) / x.shape[0]
            self.optimizer.apply_gradients([(completeGrad, x)])
        return x

    def computeKernel(self, x):
        with tf.GradientTape() as tape:
            kernelMatrix = self.kernel(x)
        return kernelMatrix, -tape.gradient(kernelMatrix, [x])[0]

    def logprobGradient(self, x):
        with tf.GradientTape() as tape:
            logprob = tf.math.log(self.targetDistribution(x))
        return tape.gradient(logprob, [x])[0]