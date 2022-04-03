import { Matrix } from './matrix';
import fs from 'fs';
import { Layer } from './networkTypes';
import { sigmoid } from './activation';

export class NeuralNetwork {
    layers: Layer[];

    lRate: number;

    constructor(
        model: { size: number; activation: Function }[],
        learningRate?: number
    ) {
        this.layers = [];

        model.forEach((layer, i) => {
            if (i === 0) {
                this.layers.push({
                    size: layer.size,
                    act: layer.activation()[0],
                    der: layer.activation()[1],

                    weights: new Matrix(layer.size, 1),
                    bias: new Matrix(layer.size, 1),

                    output: new Matrix(layer.size, 1),

                    error: new Matrix(layer.size, 1),
                    gradient: new Matrix(layer.size, 1),
                });
            } else {
                this.layers.push({
                    size: layer.size,
                    act: layer.activation()[0],
                    der: layer.activation()[1],

                    weights: new Matrix(
                        layer.size,
                        this.layers[i - 1].size
                    ).randomize(),
                    bias: new Matrix(layer.size, 1).randomize(),

                    output: new Matrix(layer.size, 1),

                    error: new Matrix(layer.size, 1),
                    gradient: new Matrix(layer.size, 1),
                });
            }
        });

        this.lRate = learningRate ?? 0.1;
    }

    predict(inputArr: number[]) {
        this.layers[0].output = Matrix.fromArray(inputArr);

        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i].output = Matrix.multiply(
                this.layers[i].weights,
                this.layers[i - 1].output
            )
                .add(this.layers[i].bias)
                .map(this.layers[i].act);
        }

        return this.layers[this.layers.length - 1].output.toArray();
    }

    train(inputArr: number[], targetArr: number[]) {
        this.layers[0].output = Matrix.fromArray(inputArr);

        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i].output = Matrix.multiply(
                this.layers[i].weights,
                this.layers[i - 1].output
            )
                .add(this.layers[i].bias)
                .map(this.layers[i].act);
        }

        const target = Matrix.fromArray(targetArr);

        // Calculate the error of output layer
        this.layers[this.layers.length - 1].error = Matrix.subtract(
            target,
            this.layers[this.layers.length - 1].output
        );

        // calculate the gradient of output layer
        this.layers[this.layers.length - 1].gradient = Matrix.map(
            this.layers[this.layers.length - 1].output,
            this.layers[this.layers.length - 1].der
        )
            .hadamard(this.layers[this.layers.length - 1].error)
            .scale(this.lRate);

        // calculate and apply the delta output layer weights
        this.layers[this.layers.length - 1].weights.add(
            Matrix.multiply(
                this.layers[this.layers.length - 1].gradient,
                Matrix.transpose(this.layers[this.layers.length - 2].output)
            )
        );

        // calculate and apply the delta output layer bias
        this.layers[this.layers.length - 1].bias.add(
            this.layers[this.layers.length - 1].gradient
        );

        // calculate the error of hidden layers
        for (let i = this.layers.length - 2; i > 0; i--) {
            // calculate the error of hidden layer
            this.layers[i].error = Matrix.multiply(
                Matrix.transpose(this.layers[i + 1].weights),
                this.layers[i + 1].error
            );

            // calculate the gradient of hidden layer
            this.layers[i].gradient = Matrix.map(
                this.layers[i].output,
                this.layers[i].der
            )
                .hadamard(this.layers[i].error)
                .scale(this.lRate);

            // calculate and apply the delta hidden layer weights
            this.layers[i].weights.add(
                Matrix.multiply(
                    this.layers[i].gradient,
                    Matrix.transpose(this.layers[i - 1].output)
                )
            );

            // calculate and apply the delta hidden layer bias
            this.layers[i].bias.add(this.layers[i].gradient);
        }
    }
}
