import numpy as np
import matplotlib.pyplot as plt


def forward_convolution(conv_w, conv_b, input):
    out_channels, in_channels, conv_width, conv_height = conv_w.shape
    _, in_width, in_height = input.shape
    
    output = np.zeros((out_channels, in_width-conv_width+1, in_height-conv_height+1))
    for x in range(in_width - conv_width + 1):
        for y in range(in_height - conv_height + 1):
            for output_channel in range(out_channels):
                output[output_channel, x, y] = np.sum(np.multiply(input[:, x:(x+conv_width), y:(y+conv_height)], conv_w[output_channel, : , : , :])) + conv_b[output_channel]
    return output
    
def backward_convolution(conv_w, conv_b, input, output_grad):
    out_channels, in_channels, conv_width, conv_height = conv_w.shape
    _, in_width, in_height = input.shape
    _, out_width, out_height = output_grad.shape
    grad_w = np.zeros_like(conv_w)
    grad_b = np.zeros_like(conv_b)
    grad_input = np.zeros_like(input)
    
    for i in range(out_channels):
        grad_b[i] += np.sum(output_grad[i])
        
    for x in range(out_width):
        for y in range(out_height):
            for out_c in range(out_channels):
                # Gradient w.r.t weights
                grad_w[out_c] += output_grad[out_c, x, y] * input[:, x:x+conv_width, y:y+conv_height]
                
                # Gradient w.r.t input
                grad_input[:, x:x+conv_width, y:y+conv_height] += output_grad[out_c, x, y] * conv_w[out_c]

    return grad_w, grad_b, grad_input
    
def forward_maxpooling(input, pool_width, pool_height):
    input_channels, input_width, input_height = input.shape

    output = np.zeros((input_channels, input_width // pool_width, input_height // pool_height))

    for x in range(0, input_width, pool_width):
        for y in range(0, input_height, pool_height):
            if x // pool_width < output.shape[1] and y // pool_height < output.shape[2]:
                output[:, x // pool_width, y // pool_height] = np.amax(input[:, x:(x + pool_width), y:(y + pool_height)], axis=(1, 2))

    return output
    
def backward_maxpooling(input, pool_width, pool_height, output_grad):
    input_channels, input_width, input_height = input.shape
    grads = np.zeros(input.shape)
    
    for x in range(0, input_width, pool_width):
        for y in range(0, input_height, pool_height):
            window = input[:, x:(x+pool_width), y:(y+pool_height)]
            # For each channel, find the location of the max within the pooling window
            flat = window.reshape(input_channels, -1)
            max_idx = np.argmax(flat, axis=1)
            for i in range(input_channels):
                # unravel index into (pool_width, pool_height)
                local_x, local_y = divmod(int(max_idx[i]), pool_height)
                if x // pool_width < output_grad.shape[1] and y // pool_height < output_grad.shape[2]:
                    grads[i, x + local_x, y + local_y] = output_grad[i, x // pool_width, y // pool_height]
    
    
    return grads

def forward_Relu(x):
    return np.maximum(0, x)
def backward_Relu(x, output_grad):
    grads = np.zeros(x.shape)
    grads[x>0] =output_grad[x>0]
    return grads

def forward_linear(w, b, input):
    return input.dot(w) + b

def backward_linear(w, b, input, output_grad):
    grad_weights = np.outer(input, output_grad)
    grad_bias = output_grad
    grad_data =np.dot(w, output_grad) ## want 50, 1
    
    return (grad_weights , grad_bias, grad_data)

def forward_softmax(x):
    x = x - np.max(x,axis=0)
    exp = np.exp(x)
    s = exp / np.sum(exp,axis=0)
    return s

def backward_softmax(x, output_grad):
    s = forward_softmax(x)
    return output_grad * s * (1-s)

def cross_entropy_loss(probabilities, labels):
    probs = np.clip(probabilities, 1e-15, 1.0)
    labels = np.asarray(labels)
    return -float(np.sum(labels * np.log(probs)))

def backward_crossentropy_loss(probabilities, labels):
    return -labels/probabilities


class CNN():
    def __init__(self, Max_poling_size=4, conv_size=3, conv_filters=2):
        self.MAX_POOL_SIZE = Max_poling_size
        self.CONVOLUTION_SIZE = conv_size
        self.CONVOLUTION_FILTERS = conv_filters
    
    def get_initial_params(self):
        

        size_after_convolution = 28 - self.CONVOLUTION_SIZE + 1
        size_after_max_pooling = size_after_convolution // self.MAX_POOL_SIZE

        num_hidden = size_after_max_pooling * size_after_max_pooling * self.CONVOLUTION_FILTERS

        return {
            'W1': np.random.normal(size = (self.CONVOLUTION_FILTERS, 1, self.CONVOLUTION_SIZE, self.CONVOLUTION_SIZE), scale=1/ np.sqrt(self.CONVOLUTION_SIZE * self.CONVOLUTION_SIZE)),
            'b1': np.zeros(self.CONVOLUTION_FILTERS),
            'W2': np.random.normal(size = (num_hidden, 10), scale = 1/ np.sqrt(num_hidden)),
            'b2': np.zeros(10)
        }
        
    
    
    def forward_prop(self, data, labels, params):
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']

        first_convolution = forward_convolution(W1, b1, data)
        first_max_pool = forward_maxpooling(first_convolution, self.MAX_POOL_SIZE, self.MAX_POOL_SIZE)
        first_after_relu = forward_Relu(first_max_pool)

        flattened = np.reshape(first_after_relu, (-1))
        
        logits = forward_linear(W2, b2, flattened)

        y = forward_softmax(logits)
        cost = cross_entropy_loss(y, labels)

        return y, cost

    def backward_prop(self, data, labels, params):
    
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']

        first_convolution = forward_convolution(W1, b1, data)
        first_max_pool = forward_maxpooling(first_convolution, self.MAX_POOL_SIZE, self.MAX_POOL_SIZE)
        first_after_relu = forward_Relu(first_max_pool)
        flattened = np.reshape(first_after_relu, (-1))
        
        logits = forward_linear(W2, b2, flattened)
        y = forward_softmax(logits)
        
        # Combined gradient for softmax + cross-entropy
        grad_logits = y - labels  # This is the simplified gradient!
        
        backward_grad_linear = backward_linear(W2, b2, flattened, grad_logits)
        grad_data = backward_grad_linear[2].reshape(first_after_relu.shape)
        backward_grad_relu = backward_Relu(first_max_pool, grad_data)
        backward_grad_maxpooliong = backward_maxpooling(first_convolution, self.MAX_POOL_SIZE, self.MAX_POOL_SIZE, backward_grad_relu)
        backward_grad_conv = backward_convolution(W1, b1, data, backward_grad_maxpooliong)
        
        return {"W1" : backward_grad_conv[0], "b1" : backward_grad_conv[1], "W2" : backward_grad_linear[0], "b2" : backward_grad_linear[1]}
    

    @staticmethod
    def nn_train(
        train_data, train_labels, dev_data, dev_labels,
        cnn_instance,  # Pass the CNN instance
        learning_rate=5.0, batch_size=16, epochs=400):
        print('Starting training with learning rate {}, batch size {}, epochs {}'.format(
            learning_rate, batch_size, epochs))
        # Get initial parameters
        params = cnn_instance.get_initial_params()

        cost_dev = []
        accuracy_dev = []
        
        for batch in range(epochs):
            print('Currently processing epoch {} / {}'.format(batch, epochs))

            # Get batch data
            batch_data = train_data[batch * batch_size:(batch + 1) * batch_size, :, :, :]
            batch_labels = train_labels[batch * batch_size: (batch + 1) * batch_size]

            # Evaluate on dev set every 100 epochs
            if batch % 100 == 0:
                output, cost = CNN.forward_prop_batch(dev_data, dev_labels, params, cnn_instance)
                cost_dev.append(sum(cost) / len(cost))
                accuracy_dev.append(CNN.compute_accuracy(output, dev_labels))

                print('Cost and accuracy', cost_dev[-1], accuracy_dev[-1])

            # Update parameters
            CNN.gradient_descent_batch(batch_data, batch_labels,
                learning_rate, params, cnn_instance)

        return params, cost_dev, accuracy_dev


    @staticmethod
    def forward_prop_batch(batch_data, batch_labels, params, cnn_instance):
        y_array = []
        cost_array = []

        for item, label in zip(batch_data, batch_labels):
            y, cost = cnn_instance.forward_prop(item, label, params)
            y_array.append(y)
            cost_array.append(cost)

        return np.array(y_array), np.array(cost_array)


    @staticmethod
    def gradient_descent_batch(batch_data, batch_labels, learning_rate, params, cnn_instance):
        total_grad = {}

        for i in range(batch_data.shape[0]):
            grad = cnn_instance.backward_prop(
                batch_data[i],
                batch_labels[i],
                params)
            
            for key, value in grad.items():
                if key not in total_grad:
                    total_grad[key] = np.zeros(value.shape)

                total_grad[key] += value

        # Update parameters
        params['W1'] = params['W1'] - learning_rate * total_grad['W1'] / batch_data.shape[0]
        params['W2'] = params['W2'] - learning_rate * total_grad['W2'] / batch_data.shape[0]
        params['b1'] = params['b1'] - learning_rate * total_grad['b1'] / batch_data.shape[0]
        params['b2'] = params['b2'] - learning_rate * total_grad['b2'] / batch_data.shape[0]


    @staticmethod
    def compute_accuracy(output, labels):
        
        correct_output = np.argmax(output, axis=1)
        correct_labels = np.argmax(labels, axis=1)

        is_correct = [a == b for a, b in zip(correct_output, correct_labels)]

        accuracy = sum(is_correct) * 1.0 / labels.shape[0]
        return accuracy


    @staticmethod
    def nn_test(data, labels, params, cnn_instance):
        
        output, cost = CNN.forward_prop_batch(data, labels, params, cnn_instance)
        accuracy = CNN.compute_accuracy(output, labels)
        return output, accuracy

