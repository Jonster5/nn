export class Matrix {
    rows: number;
    cols: number;
    data: number[][];

    constructor(rows: number, cols: number) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows)
            .fill(0)
            .map(() => Array(this.cols).fill(0));
    }

    static fromArray(arr: number[]) {
        return new Matrix(arr.length, 1).map((_, i) => arr[i] as number);
    }

    static import(m: string) {
        const data = JSON.parse(m);
        const matrix = new Matrix(data.rows, data.cols);
        matrix.data = data.data;
        return matrix;
    }

    static multiply(a: Matrix, b: Matrix) {
        if (a.cols !== b.rows) {
            console.error(`Multiplication Error: ${a.cols} !== ${b.rows}`);
        }
        return new Matrix(a.rows, b.cols).map((_, i, j) => {
            let sum = 0;
            for (let k = 0; k < a.cols; k++) {
                sum += a.data[i][k] * b.data[k][j];
            }
            return sum;
        });
    }

    static subtract(a: Matrix, b: Matrix) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            console.error(
                `Subtraction Error: ${a.rows} !== ${b.rows} || ${a.cols} !== ${b.cols}`
            );
        }
        return a.clone().map((e, i, j) => e - b.data[i][j]);
    }

    static transpose(m: Matrix) {
        const n = new Matrix(m.cols, m.rows);

        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                n.data[j][i] = m.data[i][j];
            }
        }

        return n;
    }

    transpose() {
        const d: number[][] = new Array(this.rows)
            .fill(0)
            .map(() => new Array(this.cols).fill(0));

        // Transpose the data
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                d[j][i] = this.data[i][j];
            }
        }

        // Update dimensions
        this.data = d;
        [this.rows, this.cols] = [this.cols, this.rows];

        return this;
    }

    clone() {
        return new Matrix(this.rows, this.cols).map(
            (_, i, j) => this.data[i][j]
        );
    }

    toArray() {
        let arr = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        return arr;
    }

    randomize() {
        return this.map((e) => Math.random() * 2 - 1);
    }

    add(n: Matrix | number) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                console.error(
                    `Addition Error: ${this.rows} !== ${n.rows} || ${this.cols} !== ${n.cols}`
                );
            }
            return this.map((e, i, j) => e + n.data[i][j]);
        } else {
            return this.map((e) => e + n);
        }
    }

    subtract(n: Matrix | number) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                console.error(
                    `Subtraction Error: ${this.rows} !== ${n.rows} || ${this.cols} !== ${n.cols}`
                );
            }
            return this.map((e, i, j) => e - n.data[i][j]);
        } else {
            return this.map((e) => e - n);
        }
    }

    scale(n: number) {
        return this.map((e) => e * n);
    }

    hadamard(n: Matrix) {
        // Hadamard product
        if (this.rows !== n.rows || this.cols !== n.cols) {
            console.error(
                `Hadamard product error: ${this.rows} !== ${n.rows} || ${this.cols} !== ${n.cols}`
            );
        }
        return this.map((e, i, j) => e * n.data[i][j]);
    }

    map(func: (x: number, i: number, j: number) => number) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = func(this.data[i][j], i, j);
            }
        }
        return this;
    }

    static map(a: Matrix, func: (x: number, i: number, j: number) => number) {
        return a.clone().map(func);
    }

    print() {
        console.table(this.data);
        return this;
    }

    export() {
        return JSON.stringify(this);
    }
}
