console.warn("TENSORS: ");
// Tensors

/**
 * The core datastructure of Tensorflow. 
 * The can be Scalars, Vectors and matrices.
 * They are inmutable.
 */
const shape = [2, 3]

const aTensor = tf.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape)

aTensor.print()
console.log(aTensor)

const bTensor = tf.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
bTensor.print()
console.log(bTensor)

const aScaler = tf.scalar(3)

aScaler.print()

const aTensor2 = tf.tensor2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
aTensor2.print()


const zeros = tf.zeros([3,5])
zeros.print()

const ones = tf.ones([3,5])
ones.print()