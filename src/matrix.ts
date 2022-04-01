export class Matrix {
    data: number[][];
    rows: number;
    cols: number;

    constructor(rows: number, cols: number) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows).fill(0).map(() => Array(this.cols).fill(0));
    }

    static fromArray(arr: number[]) {
        return new this(arr.length, 1).map((x, i) => arr[i]);
    }

    static import() {
        // return new this();
    }

    export() {
        return JSON.stringify(this);
    }

    randomize() {
        return this.map(() => Math.random());
    }

    add(n: Matrix | number) {
        if (typeof n === 'number') {
            return this.map(x => x + n);
        } else {
            if (this.rows !== n.rows || this.cols !== n.cols) throw 'rows and columns do not match';
            return this.map((x, i, j) => x + n.data[i][j]);
        }
    }

    subtract(n: Matrix | number) {
        if (typeof n === 'number') {
            return this.map(x => x - n);
        } else {
            if (this.rows !== n.rows || this.cols !== n.cols) throw 'rows and columns do not match';
            return this.map((x, i, j) => x - n.data[i][j]);
        }
    }

    multiply(n: Matrix | number) {
        if (typeof n === 'number') {
            return this.map(x => x * n);
        } else {
            if (this.cols !== n.rows) throw 'Columns of A must match rows of B.';

            return this.map((x, i, j) => x * n.data[i][j]);
        }
    }

    map(func: (value: number, row: number, col: number) => number) {
        for (let i = 0; i < this.data.length; i++) {
            for (let j = 0; j < this.data[i].length; j++) {
                this.data[i][j] = func(this.data[i][j], i, j);
            }
        }

        return this;
    }

    transpose() {
        return new Matrix(this.cols, this.rows).map((_, i, j) => this.data[j][i])
    }

    clone() {
        return new Matrix(this.rows, this.cols).map((_, i, j) => this.data[i][j])
    }

    toArray() {
        return [...this.data];
    }
}