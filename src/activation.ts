export class Activation {
    static sigmoid(x: number, derivative?: boolean): number {
        if (derivative) {
            return (1 / (1 + Math.exp(-x))) * (1 - 1 / (1 + Math.exp(-x)));
        } else {
            return 1 / (1 + Math.exp(-x));
        }
    }

    static tanh(x: number, derivative?: boolean): number {
        if (derivative) {
            return 1 - Math.tanh(x) * Math.tanh(x);
        } else {
            return Math.tanh(x);
        }
    }

    static relu(x: number, derivative?: boolean): number {
        if (derivative) {
            return x > 0 ? 1 : 0;
        } else {
            return x > 0 ? x : 0;
        }
    }

    static leakyRelu(x: number, derivative?: boolean): number {
        if (derivative) {
            return x > 0 ? 1 : 0.01;
        } else {
            return x > 0 ? x : 0.01 * x;
        }
    }

    static elu(x: number, derivative?: boolean): number {
        if (derivative) {
            return x > 0 ? 1 : Math.exp(x);
        } else {
            return x > 0 ? x : Math.exp(x) - 1;
        }
    }

    static binaryStep(x: number, derivative?: boolean): number {
        if (derivative) {
            return x > 0 ? 1 : 0;
        } else {
            return x > 0 ? 1 : 0;
        }
    }

    static logistic(x: number, derivative?: boolean): number {
        if (derivative) {
            return Activation.logistic(x) * (1 - Activation.logistic(x));
        } else {
            return 1 / (1 + Math.exp(-x));
        }
    }

    static arctan(x: number, derivative?: boolean): number {
        if (derivative) {
            return 1 / (x * x + 1);
        } else {
            return Math.atan(x);
        }
    }
}
