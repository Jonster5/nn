import { NeuralNetwork } from './network';

const net = new NeuralNetwork([2, 3, 1]);

const set = [
    {
        input: [0, 0],
        target: [0]
    },
    {
        input: [1, 0],
        target: [1]
    },
    {
        input: [0, 1],
        target: [1]
    },
    {
        input: [1, 1],
        target: [0]
    },

]

// for (let i = 0; i < 1000; i++) {
//     const r = Math.floor(Math.random() * 4);
//     net.train([...set[r].input], [...set[r].target]);
// }

net.train([0, 0], [0]);

console.log(net.predict([0, 0]));
console.log(net.predict([0, 1]));
console.log(net.predict([1, 0]));
console.log(net.predict([1, 1]));

net.export('output/export.json');
