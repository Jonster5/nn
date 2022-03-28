import { Layer } from './layer';
import { relu, reluPrime, sigmoid, sigmoidPrime } from './math';
import fs from 'fs';
import path from 'path';

export class NeuralNetwork {
    layers: Layer[];
    learningRate: number;
    momentum: number;

    activation: (x: number) => number;
    activationPrime: (x: number) => number;

    constructor(
        model: number[],
        rate?: number,
        momentum?: number,
        activation?: (x: number) => number,
        activationPrime?: (x: number) => number
    ) {
        this.layers = [];

        for (let i = 0; i < model.length; i++) {
            this.layers[i] = new Layer(model[i]);
        }

        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i].connect(this.layers[i - 1]);
        }

        this.learningRate = rate ?? 0.1;
        this.momentum = momentum ?? 0.1;
        this.activation = activation ?? sigmoid;
        this.activationPrime = activation ?? sigmoidPrime;
    }

    static import(path: string) {
        return new this([1]);
    }

    export(file: string) {
        const layers = this.layers.map((layer) => {
            return layer.nodes.map((node, i) => {
                return {
                    value: node.value,
                    connections: node.connections.map((c) => c.value),
                    output: node.output
                };
            });
        });

        fs.writeFileSync(path.resolve(file), JSON.stringify(layers));
    }

    predict(input: number[]) {
        this.layers[0].nodes.forEach((node, i) => {
            node.output = input[i];
        });

        this.layers.forEach((layer) => {
            layer.nodes.forEach((node) => {
                const sum = node.connections.reduce(
                    (sum, con) => sum + con.left.output * con.value,
                    0
                );

                node.output = this.activation(sum + node.value);
            });
        });

        const output = this.layers[this.layers.length - 1].nodes.map(
            (n) => n.output
        );

        return output;
    }

    train(input: number[], target: number[]) {
        this.layers[0].nodes.forEach((node, i) => {
            node.output = input[i];
        });

        this.layers.forEach((layer) => {
            layer.nodes.forEach((node) => {
                const sum = node.connections.reduce(
                    (sum, con) => sum + con.left.output * con.value,
                    0
                );

                node.output = this.activation(sum + node.value);
            });
        });

        this.layers[this.layers.length - 1].nodes.forEach((node, i) => {
            node.error = this.activationPrime(node.output) * target[i] - node.output;
            node.value += node.error * this.learningRate;

            node.connections.forEach(weight => {
                weight.value += node.error * this.learningRate;
            })
        });

        for (let i = this.layers.length - 2; i >= 0; i--) {
            this.layers[i].nodes.forEach((node, j) => {
                node.error = this.activationPrime(node.output) * this.layers[i + 1].nodes.reduce((a, b) => a * b.error, 0) * node.value; 
                node.value += node.error * this.learningRate;

                if (node.connections.length > 0) {
                    node.connections.forEach(weight => {
                        weight.value += node.error * this.learningRate;
                    })
                }
            });
        }
        
        this.layers.forEach(l => l.reset());
    }
}

/* formulas:
    feedforward:
    node output = f((i1 * w1 + i2 * w2 + ...) + bias)

    Backpropogation:
    output layer error = f'(output) * target - output

    for each layer (in reverse excluding output and input layers):
    delta error = f'(current output) * (previous layer node error * value)
    weight/bias change = value + delta error * lrate

*/
