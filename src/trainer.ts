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
                console.log('Mean Error:', error);

                if (save) console.log('saved to:', save);
                console.log();
            }

            if (error < this.goal) break;
            if (maxCycles && cycles >= maxCycles) break;
        }
    }

    demonstrate() {
        for (let i = 0; i < this.dataset.length; i++) {
            const { input, target } = this.dataset[i];

            console.log(`[ ${input} ] => [ ${this.net.predict(input)} ]`);
        }
    }

    private train() {
        let error = 0;
        for (let i = 0; i < this.cycle; i++) {
            const r = Math.floor(Math.random() * this.dataset.length);
            const { input, target } = this.dataset[r];

            this.net.train(input, target);
        }

        for (let i = 0; i < this.dataset.length; i++) {
            const { input, target } = this.dataset[i];

            error += this.net.checkError(input, target);
        }

        return error / this.cycle;
    }
}
