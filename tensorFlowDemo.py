import tensorflow as tf
import numpy as np

def trnsor_demo():
    # 使用 NumPy 生成假数据(phony data), 总共 100 个点.
    x_data = np.float32(np.random.rand(2, 100))  # 随机输入
    y_data = np.dot([0.100, 0.200], x_data) + 0.300

    # 构造一个线性模型
    #
    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b

    # 最小化方差
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 初始化变量
    init = tf.initialize_all_variables()

    # 启动图 (graph)
    sess = tf.Session()
    sess.run(init)

    # 拟合平面
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))

def tf_demo01():

    # 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
    # 加到默认图中.
    #
    # 构造器的返回值代表该常量 op 的返回值.
    matrix1 = tf.constant([[3., 3.]])

    # 创建另外一个常量 op, 产生一个 2x1 矩阵.
    matrix2 = tf.constant([[2.], [2.]])

    # 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
    # 返回值 'product' 代表矩阵乘法的结果.
    product = tf.matmul(matrix1, matrix2)

    sess=tf.Session()
    # 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
    # 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
    # 矩阵乘法 op 的输出.
    #
    # 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
    #
    # 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
    #
    # 返回值 'result' 是一个 numpy `ndarray` 对象.
    result=sess.run(product)
    print(result)

    # ==> [[ 12.]]

    # 任务完成, 关闭会话.
    sess.close()

def tf_demo03():

    # 进入一个交互式 TensorFlow 会话.
    sess=tf.InteractiveSession()
    x=tf.Variable([1.,2.])
    a=tf.constant([3.,3.])

    # 使用初始化器 initializer op 的 run() 方法初始化 'x'
    x.initializer.run()

    # 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果
    sub=tf.subtract(x,a)
    print(sub.eval())

def tf_demo04():
    a=[0.9,2.5,2.3,-4.4]
    x=tf.round(a)
    with tf.Session() as sess:
        result=sess.run(x)
    print(result)


#添加操作
def tf_demo05():
    a=tf.add(3,5)
    with tf.Session() as sess:
        print(sess.run(a))

"""
在实现上, TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU 或 GPU). 一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测. 如果检测到 GPU, TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作.

如果机器上有超过一个可用的 GPU, 除第一个外的其它 GPU 默认是不参与计算的. 为了让 TensorFlow 使用这些 GPU, 你必须将 op 明确指派给它们执行. with...Device 语句用来指派特定的 CPU 或 GPU 执行操作:
"""
def tf_demo06():
    with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='b')
        c=tf.matmul(a,b)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(c))
    sess.close()

"""
为了获取你的 operations 和 Tensor 被指派到哪个设备上运行, 用 log_device_placement 新建一个 session, 并设置为 True.
"""
def tf_demo07():
    a=tf.constant([1.0,2.0,3.0,4.0,5.0,6.0], shape=[2,3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c=tf.matmul(a,b)
    #新建session with log_device_placement并设置为True
    sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
    #运行这个op.
    print(sess.run(c))
    sess.close()

"""
手工指派设备 

如果你不想使用系统来为 operation 指派设备, 而是手工指派设备, 你可以用 with tf.device 创建一个设备环境, 这个环境下的 operation 都统一运行在环境指定的设备上.
"""
def tf_demo08():
    # 新建一个graph.
    with tf.device('/cpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # 新建session with log_device_placement并设置为True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # 运行这个op.
    print(sess.run(c))
    sess.run(c)
    sess.close()


"""
在多GPU系统里使用单一GPU

如果你的系统里有多个 GPU, 那么 ID 最小的 GPU 会默认使用. 如果你想用别的 GPU, 可以用下面的方法显式的声明你的偏好:
"""
def tf_demo09():
    # 新建一个 graph.
    with tf.device('/gpu:2'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    # 新建 session with log_device_placement 并设置为 True.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    # 运行这个 op.
    print( sess.run(c))

def tf_demo10():
    # 新建一个 graph.
    c = []
    for d in ['/gpu:2', '/gpu:3']:
        with tf.device(d):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
            c.append(tf.matmul(a, b))
    with tf.device('/cpu:0'):
        sum = tf.add_n(c)
    # 新建session with log_device_placement并设置为True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # 运行这个op.
    print(sess.run(sum))

# 5
def tf_demo11():
    a=tf.constant(2)
    b=tf.constant(3)
    x=tf.add(a,b)
    with tf.Session() as sess:
        writer=tf.summary.FileWriter('./graphs',sess.graph)
        print(sess.run(x))
    writer.close()

#[[2 4]]
def tf_demo12():
    a=tf.constant([[1,2]])
    b=tf.constant([2])
    x=tf.multiply(a,b)
    with tf.Session() as sess:
        y=sess.run(x)
    print(y)

"""
[[ 2  8 18]
 [ 6 18 36]]
"""
def tf_demo13():
    a=tf.constant([[1,2,3],[2,3,4]])
    b=tf.constant([[2,4,6],[3,6,9]])
    x = tf.multiply(a, b)
    with tf.Session() as sess:
        y = sess.run(x)
    print(y)

"""
[[8 8 8]
 [8 8 8]]
"""
def tf_demo14():
    a=tf.fill([2,3],8)
    with tf.Session() as sess:
        b=sess.run(a)
    print(b)

"""
[[0 0 0]
 [0 0 0]]
"""
def tf_demo15():
    a=tf.zeros([2,3],tf.int32)
    with tf.Session() as sess:
        b=sess.run(a)
    print(b)

"""
[[2 1 1]
 [1 1 1]
 [1 1 1]]
"""
def tf_demo16():
    a=tf.constant([2,1],shape=[3,3])
    with tf.Session() as sess:
        b = sess.run(a)
    print(b)

"""
[[2 3]
 [4 5]] [[0 2]
 [4 6]]
"""
def tf_demo17():
    a=tf.constant([2,2] ,name="a")
    b=tf.constant([[0,1],[2,3]],name="b")
    x=tf.add(a,b, name="add")
    y=tf.multiply(a,b, name="mul")
    with tf.Session() as sess:
        x,y=sess.run([x,y])
        print(x,y)


if __name__ == '__main__':
    tf_demo17()

