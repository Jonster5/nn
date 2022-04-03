import { relu, sigmoid, tanh } from './activation';
import { Matrix } from './matrix';
import { NeuralNetwork } from './network';

const net = new NeuralNetwork([
    { size: 2, activation: sigmoid },
    { size: 3, activation: sigmoid },
    { size: 1, activation: sigmoid },
]);

const set = [
    {
        input: [0, 0],
        target: [0],
    },
    {
        input: [1, 0],
        target: [1],
    },
    {
        input: [0, 1],
        target: [1],
    },
    {
        input: [1, 1],
        target: [0],
    },
];

console.log('Before training:');
console.log('[0, 0] =>', net.predict([0, 0]));
console.log('[1, 0] =>', net.predict([1, 0]));
console.log('[0, 1] =>', net.predict([0, 1]));
console.log('[1, 1] =>', net.predict([1, 1]));

for (let i = 0; i < 1000000; i++) {
    const r = Math.floor(Math.random() * 4);
    net.train(set[i % 4].input, set[i % 4].target);
}

console.log('After training:');
console.log('[0, 0] =>', net.predict([0, 0]));
console.log('[1, 0] =>', net.predict([1, 0]));
console.log('[0, 1] =>', net.predict([0, 1]));
console.log('[1, 1] =>', net.predict([1, 1]));
