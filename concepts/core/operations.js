console.warn("OPERATIONS: ");
// Operations (Ops)

/**
 * To manipulate data.
 * A wide variety of ops suitable for linear algebra and machine learning.
 * Ops return a new tensor.
 */


const d = tf.tensor2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
const d_squared = d.square();
d_squared.print();

const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);

const e_mas_f = e.add(f);
e_mas_f.print();

const e_mul_f = e.mul(f)
e_mul_f.print();

const sq_sum = e.add(f).square();
sq_sum.print()

const sq_sum_tf = tf.square(tf.add(e,f));
sq_sum_tf.print();
