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

        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i].nodes.forEach(node => {
                const sum = node.connections.reduce((s, c) => s + c.left.output * c.value, 0);
                node.output = this.activation(sum + node.value);
            });
        }

        const output = this.layers[this.layers.length - 1].nodes.map(
            (n) => n.output
        );

        return output;
    }

    train(input: number[], target: number[]) {
        this.layers[0].nodes.forEach((node, i) => {
            node.output = input[i];
        });

        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i].nodes.forEach(node => {
                const sum = node.connections.reduce((s, c) => s + c.left.output * c.value, 0);
                node.output = this.activation(sum + node.value);
            });
        }

        this.layers[2].nodes.forEach((node, i) => {
            node.error = target[i] - node.output;
        });

        this.layers[2].nodes.forEach((node, i) => {
            node.gradient = node.error * this.activationPrime(node.output) * this.learningRate;
        });

        this.layers[2].nodes.forEach((node, i) => {
            node.delta = this.layers[1].nodes.reduce((a, b) => a *= b.output, node.gradient);

        })

        this.layers[2].nodes.forEach((node, i) => {
            node.connections.forEach(c => {
                c.value += node.delta;
            });
            node.value += node.gradient;
        });

        this.layers[1].nodes.forEach((node, i) => {
            node.error = target[i] - node.output;
        });

        this.layers[1].nodes.forEach((node, i) => {
            node.gradient = node.error * this.activationPrime(node.output);
        });

        this.layers[1].nodes.forEach((node, i) => {
            node.delta = this.layers[0].nodes.reduce((a, b) => a *= b.output, node.gradient);

        })

        this.layers[1].nodes.forEach((node, i) => {
            node.connections.forEach(c => {
                c.value += node.delta;
            });
            node.value += node.gradient;
        });

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
