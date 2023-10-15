// gradient function from g9.js
function g9gradient(f, x) {
    var dim = x.length,
        f1 = f(x);
    if (isNaN(f1)) throw new Error('The gradient at [' + x.join(' ') + '] is NaN!');
    var { max, abs, min } = Math
    var tempX = x.slice(0),
        grad = Array(dim);
    for (var i = 0; i < dim; i++) {
        var delta = max(1e-6 * f1, 1e-8);
        for (var k = 0;; k++) {
            if (k == 20) throw new Error("Gradient failed at index " + i + " of [" + x.join(' ') + "]");
            tempX[i] = x[i] + delta;
            var f0 = f(tempX);
            tempX[i] = x[i] - delta;
            var f2 = f(tempX);
            tempX[i] = x[i];
            if (!(isNaN(f0) || isNaN(f2))) {
                grad[i] = (f0 - f2) / (2 * delta)
                var t0 = x[i] - delta;
                var t1 = x[i];
                var t2 = x[i] + delta;
                var d1 = (f0 - f1) / delta;
                var d2 = (f1 - f2) / delta;
                var err = min(max(abs(d1 - grad[i]), abs(d2 - grad[i]), abs(d1 - d2)), delta);
                var normalize = max(abs(grad[i]), abs(f0), abs(f1), abs(f2), abs(t0), abs(t1), abs(t2), 1e-8);
                if (err / normalize < 1e-3) break; //break if this index is done
            }
            delta /= 16
        }
    }
    return grad;
}

function almostEqual(a, b, absoluteError, relativeError) {
    const d = Math.abs(a - b);

    if (d <= absoluteError) {
        return true;
    }
    if (d <= relativeError * Math.min(Math.abs(a), Math.abs(b))) {
        return true;
    }
    return a === b;
}

function assertAllClose(a, b) {
    console.assert(a.length === b.length);
    for (let i = 0; i < a.length; i++)  {
        console.assert(almostEqual(a[i], b[i], 1e-3, 1e-3), a[i], b[i]);
    }
}

// compare g9 and tf computation of the same N dimensional gradient
function benchtf(nparams) {
    const [cx, cy] = [30.5, 42.5];

    function g9model(theta) {
        let accX = 0.;
        let accY = 0.;

        for (let i = 0; i < theta.length; i++) {
            accX += (i + 1) / theta.length * theta[i];
            accY += (theta.length - i) / theta.length * (15 - theta[i]);
        }

        return (cx - accX) ** 2 + (cy - accY) ** 2;
    }

    function tfmodel(theta) {
        return tf.tidy(() => {
            const baseX = tf.div(tf.range(1, theta.size + 1), theta.size);
            const baseY = tf.div(tf.range(theta.size, 0), theta.size);

            const ry = tf.sub(tf.scalar(15), theta);

            const accX = tf.sum(tf.mul(baseX, theta));
            const accY = tf.sum(tf.mul(baseY, ry));

            return tf.add(tf.square(tf.sub(cx, accX)), tf.square(tf.sub(cy, accY)));
        });
    }

    const theta = Array.from({ length: nparams }, () => Math.floor(Math.random() * 9));

    const g9start = window.performance.now();
    const g9grad = g9gradient(g9model, theta);
    const g9time = window.performance.now() - g9start;

    const tftime = tf.tidy(() => {
        const grads = tf.grads(tfmodel);

        const tfstart = window.performance.now();
        const tfgrads = grads([tf.tensor1d(theta)])[0].dataSync();
        const tftime = window.performance.now() - tfstart;

        assertAllClose(g9grad, tfgrads);

        return tftime;
    });

    return [g9time, tftime];
}

function benchmark(nparams) {

    const arrmean = array => array.reduce((a, b) => a + b) / array.length;
    const arrmin = array => Math.min.apply(Math, array);
    const arrmax = array => Math.max.apply(Math, array);

    let g9times = [];
    let tftimes = [];
    for (let i = 0; i < 50; i++) {
        const [g9time, tftime] = benchtf(nparams);
        g9times.push(g9time);
        tftimes.push(tftime);
    }

    console.log(`g9: ${arrmin(g9times).toFixed(3)}ms (min) | ${arrmean(g9times).toFixed(3)}ms (avg) | ${arrmax(g9times).toFixed(3)}ms (max)`);
    console.log(`tf: ${arrmin(tftimes).toFixed(3)}ms (min) | ${arrmean(tftimes).toFixed(3)}ms (avg) | ${arrmax(tftimes).toFixed(3)}ms (max)`);
}

tf.setBackend("cpu");
console.log("tfjs backend: ", tf.getBackend());