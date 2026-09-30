fn predict(x: f64, w: f64, b: f64) -> f64 {
    w * x + b
}

fn mse(xs: &[f64], ys: &[f64], w: f64, b: f64) -> f64 {
    assert_eq!(xs.len(), ys.len(), "xs and ys must have the same length");
    let n = xs.len();
    let mut sum_sq = 0.0;

    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        let y_hat = predict(x, w, b);
        let e = y - y_hat;
        sum_sq += e*e;
    }

    let n_f = n as f64;
    sum_sq / n_f
}

fn gradients(xs: &[f64], ys: &[f64], w: f64, b: f64) -> (f64, f64) {
    let n = xs.len();

    let mut sum_xe = 0.0;
    let mut sum_e = 0.0;
    
    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        let e = y - predict(x, w, b);
        sum_xe += x * e;
        sum_e += e;
    }

    let n_f = n as f64;
    let dw = -2.0 / n_f * sum_xe;
    let db = -2.0 / n_f * sum_e;
    (dw, db)
}

fn main() {
    let xs: [f64; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let ys: [f64; 10] = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];

    let lr = 0.01;
    let steps = 1000;
    let (mut w, mut b) = (0.0, 0.0);

    for step in 0..steps {
        if step % 100 == 0 {
            println!(
                "step {:>4}  loss {:.6}  w {:.4}  b {:.4}",
                step, mse(&xs, &ys, w, b), w, b
            );
        }
        let (dw, db) = gradients(&xs, &ys, w, b);
        w -= lr * dw;
        b -= lr * db;
    }

    println!("final loss {:.6} w {:.4} b {:.4}", mse(&xs, &ys, w, b), w, b);
}
