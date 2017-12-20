import tensorflow as tf

def demo01():
    a=tf.constant(2)
    b=tf.constant(3)
    x=tf.add(a,b)
    with tf.Session() as sess:
        writer=tf.summary.FileWriter('./graphs',sess.graph)
        print(sess.run(x))
    writer.close()

def demo02():
    training_op=tf.add(2,3)
    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('./graphs2', sess.graph)
        total_step = 0
        while tf.train:
            total_step += 1
            sess.run(training_op)
            if total_step % 100 == 0:
                summary_str = sess.run(merged_summary_op)
                summary_writer.add_summary(summary_str, total_step)


def demo03():
    a=2
    b=3
    x=tf.add(a,b)
    y=tf.multiply(a,b)
    useless=tf.multiply(a,x)
    z=tf.pow(a,x)
    with tf.Session() as sess:
        z=sess.run(z)
        print(z)



if __name__ == '__main__':
    demo03()