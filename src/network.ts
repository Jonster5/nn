import { Matrix } from './matrix';
import fs from 'fs';
import { ActivationFunctionName } from './networkTypes';
import { Layer } from './layer';

export class NeuralNetwork {
    layers: Layer[];

    lRate: number;
    momentum: number;

    constructor(
        model: {
            size: number;
            activation?: ActivationFunctionName;
        }[],
        learningRate?: number,
        momentum?: number
    ) {
        this.layers = [];

        model.forEach((layer, i) => {
            if (i === 0) {
                this.layers.push(
                    new Layer(layer.size, 1, layer.activation ?? 'sigmoid')
                );
            } else {
                this.layers.push(
                    new Layer(
                        layer.size,
                        model[i - 1].size,
                        layer.activation ?? 'sigmoid'
                    )
                );
            }
        });

        this.lRate = learningRate ?? 0.1;
        this.momentum = momentum ?? 0.1;
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
        this.layers[this.layers.length - 1].weights
            .add(
                Matrix.multiply(
                    this.layers[this.layers.length - 1].gradient,
                    Matrix.transpose(this.layers[this.layers.length - 2].output)
                )
            )
            .add(
                Matrix.multiply(
                    this.layers[this.layers.length - 1].pGradient,
                    Matrix.transpose(this.layers[this.layers.length - 2].output)
                ).scale(this.momentum)
            );

        // calculate and apply the delta output layer bias
        this.layers[this.layers.length - 1].bias
            .add(this.layers[this.layers.length - 1].gradient)
            .add(
                this.layers[this.layers.length - 1].pGradient.scale(
                    this.momentum
                )
            );

        // update the output layer's previous gradient
        this.layers[this.layers.length - 1].pGradient =
            this.layers[this.layers.length - 1].gradient.clone();

        // return if there are no hidden layers
        if (this.layers.length < 3) {
            return Math.abs(
                this.layers[this.layers.length - 1].error
                    .toArray()
                    .reduce((a, b) => a + b, 0) /
                    this.layers[this.layers.length - 1].error.toArray().length
            );
        }

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
            this.layers[i].weights
                .add(
                    Matrix.multiply(
                        this.layers[i].gradient,
                        Matrix.transpose(this.layers[i - 1].output)
                    )
                )
                .add(
                    Matrix.multiply(
                        this.layers[i].pGradient,
                        Matrix.transpose(this.layers[i - 1].output)
                    ).scale(this.momentum)
                );

            // calculate and apply the delta hidden layer bias
            this.layers[i].bias
                .add(this.layers[i].gradient)
                .add(this.layers[i].pGradient.scale(this.momentum));

            // update the previous gradient
            this.layers[i].pGradient = this.layers[i].gradient.clone();
        }

        // return mean error of output layer
        return Math.abs(
            this.layers[this.layers.length - 1].error
                .toArray()
                .reduce((a, b) => a + b, 0) /
                this.layers[this.layers.length - 1].error.toArray().length
        );
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
            momentum: this.momentum,
        };

        fs.writeFileSync(path, JSON.stringify(model));
    }

    static import(path: string) {
        const model = JSON.parse(fs.readFileSync(path, 'utf8'));

        const nn = new NeuralNetwork(
            model.layers,
            model.learningRate,
            model.momentum
        );

        nn.layers.forEach((layer, i) => {
            layer.weights = Matrix.import(model.layers[i].weights);
            layer.bias = Matrix.import(model.layers[i].bias);
        });

        return nn;
    }
}
