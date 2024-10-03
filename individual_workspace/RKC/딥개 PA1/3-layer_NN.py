
import numpy as np

class linear():
    def __init__(self, input_dim, output_dim):
        self.weight = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim)
    
    def forward(self, input):
        return np.dot(input, self.weight) + self.bias
    
    def backward(self, input, output_gradient, learning_rate): # Backpropagation
        # input: 뉴런에 들어가는 값
        # output_gradient: 다음 뉴런의 기울기
        input_gradient = np.dot(output_gradient, self.weight) # 다음 layer에 넘겨줄 gradient
        weight_gradient = np.dot(input, output_gradient) # 업데이트 할 weight gradient
        bias_gradient = np.sum(output_gradient, axis=0)
        
        self.weight -= learning_rate*weight_gradient
        self.bias -= learning_rate*bias_gradient
        
        return input_gradient
    
class ReLU():
    def forward(self, input):
        return np.maximum(0, input)
    
    def backward(self, input, output_gradient):
        grad = input > 0
        return grad * output_gradient
    
def SoftMax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def CrossEntropyLoss(y, y_pred):
    c = 1e-10
    if y_pred < c:
        y_pred = c
    elif y_pred > 1-c:
        y_pred = 1-c
    return -np.sum(y * np.log(y_pred))
    
class NN_3_layer():
    def __init__(self, in_feature, hid_feature, out_feature):
        self.inputs = [[]*5]
        self.layer1 = linear(in_feature, hid_feature)
        self.relu1 = ReLU()
        self.layer2 = linear(hid_feature, hid_feature)
        self.relu2 = ReLU()
        self.layer3 = linear(hid_feature, out_feature)
    
    def forward(self, input):
        self.inputs[0] = input
        self.inputs[1] = linear.forward(self.inputs[0])
        self.inputs[2] = ReLU.forward(self.inputs[1])
        self.inputs[3] = linear.forward(self.inputs[2])
        self.inputs[4] = ReLU.forward(self.inputs[3])
        self.inputs[5] = linear.forward(self.inputs[4])
        output = SoftMax(self.inputs[5])
        return output
    
    def backward(self, output_gradient):
        inputs = self.inputs
        output_gradient = linear.backward(inputs[4], output_gradient)
        output_gradient = ReLU.backward(inputs[3], output_gradient)
        output_gradient = linear.backward(inputs[2], output_gradient)
        output_gradient = ReLU.backward(inputs[1], output_gradient)
        output_gradient = linear.backward(inputs[0], output_gradient)