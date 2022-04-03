export function sigmoid() {
    return [(x: number) => 1 / (1 + Math.exp(-x)), (y: number) => y * (1 - y)];
}

export function tanh() {
    return [(x: number) => Math.tanh(x), (y: number) => 1 - y * y];
}

export function relu() {
    return [(x: number) => Math.max(0, x), (y: number) => (y > 0 ? 1 : 0)];
}
