import tensorflow as tf
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# y=softmax(Wx+b)
x = tf.placeholder(tf.float32, [None , 784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros(10))

y=tf.nn.softmax(tf.matmul(x,W)+b)

y_=tf.placeholder("float",[None ,10])
#“交叉熵”（cross-entropy）
cross_entropy=-tf.reduce_sum(y_*tf.log(y))

"""
反向传播算法(backpropagation algorithm)来有效地确定你的变量是如何影响你想要最小化的那个成本值的。然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。
"""
"""
TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动
"""
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#现在，我们已经设置好了我们的模型。在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：
init=tf.initialize_all_variables()

#现在我们可以在一个Session里面启动我们的模型，并且初始化变量：
sess=tf.Session()
sess.run(init)

#然后开始训练模型，这里我们让模型循环训练1000次！
for i in range(1000):
    batch_xs, batch_ys=mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#首先让我们找出那些预测正确的标签。tf.argmax 是一个非常有用的函数，
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#最后，我们计算所学习到的模型在测试数据集上面的正确率。
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_ : mnist.test.labels}))



