import { Layer } from './layer';
import { relu, reluPrime, sigmoid, sigmoidPrime } from './math';
import fs from 'fs';

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

    export(path: string) {
        const layers = this.layers.map((layer) => {
            return layer.nodes.map((node, i) => {
                return {
                    value: node.value,
                    connections: node.connections.map((c) => c.value),
                };
            });
        });

        console.log(layers);
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
        this.predict(input);

        this.layers[this.layers.length - 1].nodes.forEach((node, i) => {
            node.error = target[i] - node.output;
        });

        // return this.predict(input);
    }
}

/* formulas:
    feedforward:
    layer output = f((i1 * w1 + i2 * w2 + ...) + bias)

    Backpropogation:
    output error = target - output
    hidden error = output error * output layer weights * f'(hidden output)

    delta = (learning rate * value * error) + (momentum * previous delta)

*/
