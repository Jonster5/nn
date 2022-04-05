import { NeuralNetwork } from './network';

export class KnightsTour {
    board: number[][];
    brain: NeuralNetwork;

    constructor() {
        this.board = new Array(8).fill(0).map(() => new Array(8).fill(0));
        this.board[0][0] = 1;

        this.brain = NeuralNetwork.import('output/knight.net.json');
        this.display(1);
    }

    display(n: number, confidence: number = 100) {
        console.log(`Move: ${n}`, `Confidence: ${confidence}%`);
        this.board.forEach((row) => {
            let line = '';
            row.forEach((col) => {
                if (col < 10) {
                    line += `0${col} `;
                } else {
                    line += `${col} `;
                }
            });
            console.log(line);
        });
        console.log();
    }

    nextMove() {
        let r = 0;
        let c = 0;

        for (let i = 0, m = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                if (this.board[i][j] > m) {
                    m = this.board[i][j];
                    r = i;
                    c = j;
                }
            }
        }

        // list of moves counterclockwise from bottom right
        const moves = [
            [1, 2], // down right
            [2, 1], // down left
            [2, -1], // up left
            [1, -2], // up right
            [-1, -2], // up left
            [-2, -1], // down left
            [-2, 1], // down right
        ];

        // convert board into a single array
        const b = this.board.reduce((acc, row) => {
            return acc.concat(row);
        }, []);

        console.log(b);

        const output = this.brain.predict(b);

        // find the highest probability
        let index = 0;
        for (let i = 0, m = 0; i < output.length; i++) {
            if (output[i] > m) {
                m = output[i];
                index = i;
            }
        }

        const move = moves[index];

        this.board[r + move[0]][c + move[1]] = this.board[r][c] + 1;

        this.display(
            this.board[r + move[0]][c + move[1]],
            parseFloat(output[index].toFixed(4)) * 100
        );
    }
}
