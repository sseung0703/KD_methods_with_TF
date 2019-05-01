import tensorflow as tf
import numpy as np

def SVD(X, n, name = None):
    with tf.variable_scope(name):
        sz = X.get_shape().as_list()
        if len(sz)>2:
            x = tf.reshape(X,[sz[0],sz[1]*sz[2],sz[3]])
            n = min(n, sz[1]*sz[2], sz[3])
        else:
            x = tf.reshape(X,[sz[0],1,-1])
            n = 1

        sz = x.get_shape().as_list()
        with tf.device('CPU'):
            g = tf.get_default_graph()
            with g.gradient_override_map({"Svd": "Svd_"}):
                s,u,v = tf.svd(x,full_matrices=False)
                
        s = tf.slice(s,[0,0],[-1,n])
        s = removenan(s)
        s = s/tf.sqrt(tf.reduce_sum(tf.square(s),1,keepdims=True)+1e-3)
        U = tf.slice(u,[0,0,0],[-1,-1,n])
        V = tf.slice(v,[0,0,0],[-1,-1,n])
        
        V = removenan(V)
        U = removenan(U)
        
        V /= tf.sqrt(tf.reduce_sum(tf.square(V),1,keepdims=True)+1e-3)
        U /= tf.sqrt(tf.reduce_sum(tf.square(U),1,keepdims=True)+1e-3)
        
        return s, U, V

def Align_rsv(x, y, x_s, y_s, k):
    x_sz = x.get_shape().as_list()
    x_ = tf.transpose(x,[0,2,1])
    x_s_ = tf.transpose(x_s,[0,2,1])
    
    x_temp = []; x_s_temp =[]
    r = tf.constant(np.array( list(range(x_sz[0])) ).reshape(-1,1,1),dtype=tf.int32)
    cosine = tf.matmul(x, y, transpose_a=True)
    index = tf.expand_dims(tf.cast(tf.argmax(tf.abs(cosine),1),tf.int32),-1)
    for i in range(k):
        idx = tf.slice(index,[0,i,0],[-1,1,-1])
        idx = tf.concat([r, idx],2)
        x_temp.append(tf.gather_nd(x_, idx ))
        x_s_temp.append(tf.gather_nd(x_s_, idx))
    
    x = tf.transpose(tf.concat(x_temp,1),[0,2,1])
    x_s = tf.transpose(tf.concat(x_s_temp,1),[0,2,1])
    
    cosine = tf.expand_dims(tf.matrix_diag_part(tf.matmul(x, y, transpose_a=True)),1)
    x   *= tf.sign(cosine)
    x_s *= tf.sign(cosine)
    
    return x, x_s, y

def removenan(x):
    return tf.where(tf.is_finite(x), x,tf.zeros_like(x))
def msym(X):
    return (X+tf.matrix_transpose(X))

@tf.RegisterGradient('Svd_')
def gradient_svd(op, ds, dU, dV):
    s, U, V = op.outputs

    u_sz = dU.get_shape().as_list()[1]
    B, v_sz = dV.get_shape().as_list()[:2]

    S = tf.matrix_diag(s)
    s_2 = tf.square(s)

    eye = tf.expand_dims(tf.eye(min(u_sz, v_sz)),0) 
    k = (1 - eye)/(tf.expand_dims(s_2,2)-tf.expand_dims(s_2,1) + eye)
    KT = tf.matrix_transpose(k)
    KT = removenan(KT)

    if u_sz<v_sz:
        U, V = (V, U); dU, dV = (dV, dU)
        D = tf.matmul(dU,tf.matrix_diag(1/(s+1e-8)))
        US = tf.matmul(U,S)
    
        grad = tf.matmul(D, V, transpose_b=True)\
              +tf.matmul(tf.matmul(U,tf.matrix_diag(tf.matrix_diag_part(-tf.matmul(U,D,transpose_a=True)))), V, transpose_b=True)\
              +tf.matmul(2*tf.matmul(US, msym(KT*(tf.matmul(V,-tf.matmul(V,tf.matmul(D,US,transpose_a=True)),transpose_a=True)))),V,transpose_b=True)
        grad = tf.matrix_transpose(grad)

    else:
        D = tf.matmul(dU,tf.matrix_diag(1/(s+1e-8)))
        US = tf.matmul(U,S)
        grad = tf.matmul(2*tf.matmul(US, msym(KT*(tf.matmul(V,dV,transpose_a=True))) ),V,transpose_b=True)
    return [grad]

