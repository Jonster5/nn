import { NeuralNetwork } from './network';
import fs from 'fs';

export class Trainer {
    net: NeuralNetwork;
    cycle: number;
    goal: number;
    dataset: { input: number[]; target: number[] }[];

    constructor(
        net: NeuralNetwork,
        dataset: { input: number[]; target: number[] }[] | string,
        goal: number,
        cycle?: number
    ) {
        this.net = net;
        this.cycle = cycle ?? 10000;
        this.goal = goal;
        if (typeof dataset === 'string') {
            this.dataset = JSON.parse(fs.readFileSync(dataset, 'utf8'));
        } else {
            this.dataset = dataset;
        }
    }

    start(props: { maxCycles?: number; verbose?: boolean; save?: string }) {
        const { maxCycles, verbose, save } = props;
        let cycles = 0;

        while (true) {
            const error = this.train();
            if (save) this.net.export(save);
            cycles++;

            if (verbose) {
                console.log('Cycle:', cycles);
                console.log(
                    'Error:',
                    parseFloat((error * 100).toFixed(10)),
                    '%'
                );

                if (save) console.log('saved to:', save);
                console.log();
            }

            if (error < this.goal) break;
            if (maxCycles && cycles >= maxCycles) break;
        }
    }

    demonstrate(num: number) {
        console.log();
        if (num > this.dataset.length) num = this.dataset.length;
        for (let i = 0; i < num; i++) {
            const r = Math.floor(Math.random() * this.dataset.length);
            const { input, target } = this.dataset[r];

            console.log(
                `[ ${input.join(', ')} ] -> [ ${this.net
                    .predict(input)
                    .map((x) => x.toFixed(4))
                    .join(', ')} ] | [ ${target.join(', ')} ]`
            );
        }
        console.log();
    }

    private train() {
        let error = 0;
        for (let i = 0; i < this.cycle; i++) {
            const r = Math.floor(Math.random() * this.dataset.length);
            const { input, target } = this.dataset[r];

            error += this.net.train(input, target);
        }

        return error / this.cycle;
    }
}
