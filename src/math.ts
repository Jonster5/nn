export function sigmoid(x: number) {
    return 1 / (1 + Math.exp(-x));
}

export function sigmoidPrime(x: number) {
    return sigmoid(x) * (1 - sigmoid(x));
}

export function tanh(x: number) {
    return Math.tanh(x);
}

export function tanhPrime(x: number) {
    return 1 - Math.tanh(x) ** 2;
}

export function relu(x: number) {
    return x > 0 ? x : 0;
}

export function reluPrime(x: number) {
    return x > 0 ? 1 : 0;
}
