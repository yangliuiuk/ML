def gi(m1, m2):
    n = len(m1)
    lr = 1e-1
    steps = 100
    sm = 1e-10
    r = 1

    n = m1.shape[0]
    
    s = tf.Variable(np.zeros([n,n], dtype='float32'), trainable=True)

    label = np.zeros([n,n], dtype='float32')

    a = np.ones([n,n], dtype='float32')
    
    optimizer = tf.keras.optimizers.Adam(lr)

    print('start')

    t1 = time.time()
    
    for ni in range(n+1):
        
        for step in range(steps):
            with tf.GradientTape() as tape:

                p = tf.nn.softmax(s, axis=1)

                loss_con = tf.reduce_sum((m1 - tf.matmul(tf.matmul(p, m2), tf.transpose(p))) ** 2)
                loss_label = -tf.reduce_sum(label * tf.math.log(p + sm))
                
                loss =  loss_con + loss_label * r
                
                grads = tape.gradient(loss, [s])
                optimizer.apply_gradients(zip(grads, [s]))

        print(ni, loss_con.numpy(), loss_label.numpy())

        pi = tf.one_hot(tf.argmax(p, axis=-1), n)

        loss_con_pi = tf.reduce_sum((m1 - tf.matmul(tf.matmul(pi, m2), tf.transpose(pi))) ** 2)

        loss_perm_pi = tf.reduce_sum((tf.matmul(pi, tf.transpose(pi)) - tf.linalg.diag(tf.ones(n)))**2)

        if loss_perm_pi == 0 and loss_con_pi == 0:
            return True

        i, j = np.unravel_index(np.argmax(p * a), p.shape)
        a[i] = 0
        label[i][j] = 1
    
    return False
