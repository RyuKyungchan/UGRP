
import numpy as np

class linear():
    def forward(self, input):
        return np.dot(input, self.weight) + self.bias
    
    def backward(self, input, output_gradient): # Backpropagation
        input_gradient = np.dot(output_gradient, self.weight) # 다음 layer에 넘겨줄 gradient
        weight_gradient = np.dot(input, output_gradient) # 업데이트 할 weight gradient
        
        weights = weights - learning_rate*weight_gradient
        #bias = bias - learning_rate*bias_gradient
        
        return input_gradient
    
class ReLU():
    def forward(input):
        return max(0, input)
    
    def backward(input, output_gradient):
        grad = input > 0
        return grad * output_gradient
    
class NN3():
    def forward(self, input):
        self.inputs[0] = input
        self.inputs[1] = linear.forward(self.inputs[0])
        self.inputs[2] = ReLU.forward(self.inputs[1])
        self.inputs[3] = linear.forward(self.inputs[2])
        self.inputs[4] = ReLU.forward(self.inputs[3])
        output = linear.forward(self.inputs[4])
        return output
    
    def backward(self, output_gradient):
        inputs = self.inputs
        output_gradient = linear.backward(inputs[4], output_gradient)
        output_gradient = ReLU.backward(inputs[3], output_gradient)
        output_gradient = linear.backward(inputs[2], output_gradient)
        output_gradient = ReLU.backward(inputs[1], output_gradient)
        output_gradient = linear.backward(inputs[0], output_gradient)