import { Matrix } from './matrix';
import fs from 'fs';
import { ActivationFunctionName, Layer } from './networkTypes';

export class NeuralNetwork {
    layers: Layer[];

    lRate: number;

    constructor(
        model: {
            size: number;
            activation?: ActivationFunctionName;
        }[],
        learningRate?: number
    ) {
        this.layers = [];

        model.forEach((layer, i) => {
            if (i === 0) {
                this.layers.push({
                    size: layer.size,
                    activation: NeuralNetwork[layer.activation ?? 'sigmoid'],
                    aName: layer.activation ?? 'sigmoid',

                    weights: new Matrix(layer.size, 1),
                    bias: new Matrix(layer.size, 1),

                    output: new Matrix(layer.size, 1),

                    error: new Matrix(layer.size, 1),
                    gradient: new Matrix(layer.size, 1),
                });
            } else {
                this.layers.push({
                    size: layer.size,
                    activation: NeuralNetwork[layer.activation ?? 'sigmoid'],
                    aName: layer.activation ?? 'sigmoid',

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
                .map((x) => this.layers[i].activation(x));
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
                .map((x) => this.layers[i].activation(x));
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
            (x) => this.layers[this.layers.length - 1].activation(x, true)
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
            this.layers[i].gradient = Matrix.map(this.layers[i].output, (x) =>
                this.layers[i].activation(x, true)
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

    checkError(inputArr: number[], targetArr: number[]) {
        const output = this.predict(inputArr);

        const error = targetArr
            .map((x, i) => Math.abs(x - output[i]))
            .reduce((a, b) => a + b);

        return error;
    }

    export(path: string) {
        const model = {
            layers: this.layers.map((layer) => ({
                size: layer.size,
                activation: layer.aName,
                weights: layer.weights.export(),
                bias: layer.bias.export(),
            })),
            learningRate: this.lRate,
        };

        fs.writeFileSync(path, JSON.stringify(model));
    }

    static import(path: string) {
        const model = JSON.parse(fs.readFileSync(path, 'utf8'));

        const nn = new NeuralNetwork(model.layers, model.learningRate);

        nn.layers.forEach((layer, i) => {
            layer.weights = Matrix.import(model.layers[i].weights);
            layer.bias = Matrix.import(model.layers[i].bias);
        });

        return nn;
    }

    static sigmoid(x: number, derivative?: boolean) {
        if (derivative) {
            return x * (1 - x);
        } else {
            return 1 / (1 + Math.exp(-x));
        }
    }

    static tanh(x: number, derivative?: boolean) {
        if (derivative) {
            return 1 - x * x;
        } else {
            return Math.tanh(x);
        }
    }

    static relu(x: number, derivative?: boolean) {
        if (derivative) {
            return x > 0 ? 1 : 0;
        } else {
            return x > 0 ? x : 0;
        }
    }

    static leakyRelu(x: number, derivative?: boolean) {
        if (derivative) {
            return x > 0 ? 1 : 0.01;
        } else {
            return x > 0 ? x : 0.01 * x;
        }
    }

    static binaryStep(x: number, derivative?: boolean) {
        if (derivative) {
            return x > 0 ? 1 : 0;
        } else {
            return x > 0 ? 1 : 0;
        }
    }
}
