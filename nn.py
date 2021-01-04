import sys
import os.path
import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Node:
    def __init__(self):
        self.weight = []
        self.upstream = []
        self.out = 0
        self.delta = 0

    def forward(self):
        total = 0
        for i in range(len(self.upstream)):
            total += self.weight[i] * self.upstream[i].out
        self.out = sigmoid(total)

    def backward(self):
        self.delta *= self.out * (1 - self.out)
        for i in range(len(self.upstream)):
            self.upstream[i].delta += self.weight[i] * self.delta

    def set_delta(self, target):
        self.delta = target - self.out

    def update(self, rate):
        for i in range(len(self.upstream)):
           self.weight[i] += rate * self.delta * self.upstream[i].out
        self.delta = 0


class Layer:
    def __init__(self, size):
        self.nodes = []
        for _ in range(size):
            self.nodes.append(Node())

    def connect_upstream(self, up_layer):
        bias = Node()
        bias.out = 1
        for node in self.nodes:
            for up in up_layer.nodes:
                node.upstream.append(up)
            node.upstream.append(bias)
            for _ in node.upstream:
                node.weight.append(random.random()-0.5)

    def set_input(self, inputs):
        for i in range(len(inputs)):
            self.nodes[i].out = inputs[i]
            self.nodes[i].delta = 0

    def set_target(self, targets):
        for i in range(len(targets)):
            self.nodes[i].set_delta(targets[i])

    def forward(self):
        for node in self.nodes:
            node.forward()

    def backward(self):
        for node in self.nodes:
            node.backward()

    def update(self, rate):
        for node in self.nodes:
            node.update(rate)


class NN:
    def __init__(self):
        self.layers = [Layer(256), Layer(10), Layer(3)]
        self.layers[1].connect_upstream(self.layers[0])
        self.layers[2].connect_upstream(self.layers[1])

    def forward(self, inputs):
        self.layers[0].set_input(inputs)
        self.layers[1].forward()
        self.layers[2].forward()
        max_out = 0
        for i in range(3):
            if self.layers[2].nodes[i].out \
                    >= self.layers[2].nodes[max_out].out:
                max_out = i
        return max_out

    def backward(self, targets, rate):
        self.layers[2].set_target(targets)
        self.layers[2].backward()
        self.layers[1].backward()
        self.layers[2].update(rate)
        self.layers[1].update(rate)

    def target(self, targets):
        max_target = 0
        for i in range(len(targets)):
            if targets[i] >= targets[max_target]:
                max_target = i
        return max_target

    def test(self, inputs, targets):
        max_out = self.forward(inputs)
        max_target = self.target(targets)
        return max_out == max_target

    def accuracy(self, inputs_all, targets_all):
        num_correct = 0
        for i in range(len(inputs_all)):
            if self.test(inputs_all[i], targets_all[i]):
                num_correct += 1
        return num_correct / len(inputs_all)

    def epoch(self, inputs_all, targets_all, rate):
        for i in range(len(inputs_all)):
            self.forward(inputs_all[i])
            self.backward(targets_all[i], rate)

    def save(self, filename):
        nn_file = open(filename, 'w')
        for node in self.layers[1].nodes:
            for w in node.weight:
                nn_file.write(str(w)+' ')
            nn_file.write('\n')
        for node in self.layers[2].nodes:
            for w in node.weight:
                nn_file.write(str(w)+' ')
            nn_file.write('\n')
        nn_file.close()

    def load(self, filename):
        nn_file = open(filename, 'r')
        content = nn_file.read()
        lines = content.splitlines()
        for i in range(10):
            items = lines[i].split()
            for j in range(len(items)):
                self.layers[1].nodes[i].weight[j] = float(items[j])
        for i in range(3):
            items = lines[10+i].split()
            for j in range(len(items)):
                self.layers[2].nodes[i].weight[j] = float(items[j])
        nn_file.close()


def main():
    if len(sys.argv) != 3:
        sys.exit('Usage: python nn.py [input file] [output file]')
    in_filename = sys.argv[1]
    out_filename = sys.argv[2]
    in_file = open(in_filename, 'r')
    in_content = in_file.read()
    in_lines = in_content.splitlines()
    inputs = []
    targets = []
    for line in in_lines:
        items = line.split()
        if line[0] == '#':
            continue
        if len(items) == 256 + 3:
            inputs.append(items[0:256])
            targets.append(items[256:259])
        elif len(items) == 256:
            inputs.append(items[0:256])
        else:
            sys.exit('Error reading input file')

    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            inputs[i][j] = float(inputs[i][j])

    for i in range(len(targets)):
        for j in range(len(targets[i])):
            targets[i][j] = float(targets[i][j])

    nn = NN()

    nn_filename = 'nn.txt'
    if os.path.isfile(nn_filename) and len(targets) == 0:
        nn.load(nn_filename)

    if len(targets) != 0:
        for i in range(50):
            print('Epoch: '+str(i)+'...')
            nn.epoch(inputs, targets, 0.5)
            accuracy = nn.accuracy(inputs, targets)
            print('Accuracy: '+str(accuracy))
            if accuracy > 0.9:
                break
        nn.save(nn_filename)

    out_file = open(out_filename, 'w')
    if len(targets) != 0:
        out_file.write('my_output target\n')
    labels = ['1', '8', '9']
    num_correct = 0
    for i in range(len(inputs)):
        out = nn.forward(inputs[i])
        if len(targets) != 0:
            target = nn.target(targets[i])
            if out == target:
                num_correct += 1
        out_file.write(labels[out])
        if len(targets) != 0:
            out_file.write(' '+labels[target]+' ')
        out_file.write('\n')

    if len(targets) != 0:
        out_file.write('Accuracy: '+str(num_correct)+'/'+str(len(inputs)))
        out_file.write(' = %.2f' % (num_correct / len(inputs)))
        out_file.write('%\n')

    out_file.close()


main()

