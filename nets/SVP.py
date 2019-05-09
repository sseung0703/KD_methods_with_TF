import tensorflow as tf
import numpy as np
from tensorflow.python.framework import function

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

    shape = tf.shape(dU)
    u_sz = tf.reduce_sum(tf.slice(shape,[1],[1]))
    
    shape = tf.shape(dV)
    v_sz = tf.reduce_sum(tf.slice(shape,[1],[1]))

    shape = tf.shape(ds)
    s_sz = tf.reduce_sum(tf.slice(shape,[1],[1]))

    S = tf.matrix_diag(s)
    s_2 = tf.square(s)

    eye = tf.expand_dims(tf.eye(s_sz),0) 
    k = (1 - eye)/(tf.expand_dims(s_2,2)-tf.expand_dims(s_2,1) + eye)
    KT = tf.matrix_transpose(k)
    KT = removenan(KT)

    def left_grad(U,S,V,dU,dV):
        U, V = (V, U); dU, dV = (dV, dU)
        D = tf.matmul(dU,tf.matrix_diag(1/(s+1e-8)))
        US = tf.matmul(U,S)
    
        grad = tf.matmul(D, V, transpose_b=True)\
              +tf.matmul(tf.matmul(U,tf.matrix_diag(tf.matrix_diag_part(-tf.matmul(U,D,transpose_a=True)))), V, transpose_b=True)\
              +tf.matmul(2*tf.matmul(US, msym(KT*(tf.matmul(V,-tf.matmul(V,tf.matmul(D,US,transpose_a=True)),transpose_a=True)))),V,transpose_b=True)
        grad = tf.matrix_transpose(grad)
        return grad

    def right_grad(U,S,V,dU,dV):
        US = tf.matmul(U,S)
        grad = tf.matmul(2*tf.matmul(US, msym(KT*(tf.matmul(V,dV,transpose_a=True))) ),V,transpose_b=True)
        return grad
    
    grad = tf.cond(tf.greater(v_sz, u_sz), lambda : left_grad(U,S,V,dU,dV), 
                                           lambda : right_grad(U,S,V,dU,dV))
    
    return [grad]


def SVD_eid(X, n, name = None):
    with tf.variable_scope(name):
        sz = X.get_shape().as_list()
        if len(sz)>2:
            x = tf.reshape(X,[sz[0],sz[1]*sz[2],sz[3]])
            n = min(n, sz[1]*sz[2], sz[3])
        else:
            x = tf.reshape(X,[sz[0],1,-1])
            n = 1
            
        B, HW, D = x.get_shape().as_list()

        x_ = tf.stop_gradient(x)
        if HW/D < 3/2  and 2/3 < HW/D:
            with tf.device('CPU'):
                g = tf.get_default_graph()
                with g.gradient_override_map({"Svd": "Svd_"}):
                    s,u,v = tf.svd(x_,full_matrices=False)

        elif HW < D:
            xxt = tf.matmul(x_,x_,transpose_b = True)
            with tf.device('CPU'):
                _,u_svd,_ = tf.svd(xxt,full_matrices=False)

            v_svd = tf.matmul(x_, u_svd, transpose_a = True)
            s_svd = tf.sqrt(tf.reduce_sum(tf.square(v_svd),1))
            v_svd = removenan(v_svd/tf.expand_dims(s_svd,1))
            
            s,u,v = SVD_grad_map(x,s_svd,u_svd,v_svd)
            s = tf.reshape(s,[B,   min(HW,D)])
            u = tf.reshape(u,[B,HW,min(HW,D)])
            v = tf.reshape(v,[B, D,min(HW,D)])
#            
        else:
            xtx = tf.matmul(x_,x_,transpose_a = True)
            with tf.device('CPU'):
                _, v_svd = tf.linalg.eigh(xtx)
            v_svd = tf.reshape(tf.image.flip_left_right(tf.reshape(v_svd,[B,D,D,1])),[B,D,D])

            u_svd = tf.matmul(x_, v_svd)
            s_svd = tf.sqrt(tf.reduce_sum(tf.square(u_svd),1))
            u_svd = removenan(u_svd/tf.expand_dims(s_svd,1))

            s,u,v = SVD_grad_map(x,s_svd,u_svd,v_svd)
            s = tf.reshape(s,[B,   min(HW,D)])
            u = tf.reshape(u,[B,HW,min(HW,D)])
            v = tf.reshape(v,[B, D,min(HW,D)])

        s = tf.slice(s,[0,0],[-1,n])
        U = tf.slice(u,[0,0,0],[-1,-1,n])
        V = tf.slice(v,[0,0,0],[-1,-1,n])
        
        s /= tf.sqrt(tf.reduce_sum(tf.square(s),1,keepdims=True)+1e-3)
        V /= tf.sqrt(tf.reduce_sum(tf.square(V),1,keepdims=True)+1e-3)
        U /= tf.sqrt(tf.reduce_sum(tf.square(U),1,keepdims=True)+1e-3)
        
        return s, U, V
    
def gradient_eid(op, ds, dU, dV):
    s, U, V = op.outputs

    shape = tf.shape(dU)
    u_sz = tf.reduce_sum(tf.slice(shape,[1],[1]))
    
    shape = tf.shape(dV)
    v_sz = tf.reduce_sum(tf.slice(shape,[1],[1]))

    shape = tf.shape(ds)
    s_sz = tf.reduce_sum(tf.slice(shape,[1],[1]))

    S = tf.matrix_diag(s)
    s_2 = tf.square(s)

    eye = tf.expand_dims(tf.eye(s_sz),0) 
    k = (1 - eye)/(tf.expand_dims(s_2,2)-tf.expand_dims(s_2,1) + eye)
    KT = tf.matrix_transpose(k)
    KT = removenan(KT)

    def left_grad(U,S,V,dU,dV):
        U, V = (V, U); dU, dV = (dV, dU)
        D = tf.matmul(dU,tf.matrix_diag(1/(s+1e-8)))
        US = tf.matmul(U,S)
    
        grad = tf.matmul(D, V, transpose_b=True)\
              +tf.matmul(tf.matmul(U,tf.matrix_diag(tf.matrix_diag_part(-tf.matmul(U,D,transpose_a=True)))), V, transpose_b=True)\
              +tf.matmul(2*tf.matmul(US, msym(KT*(tf.matmul(V,-tf.matmul(V,tf.matmul(D,US,transpose_a=True)),transpose_a=True)))),V,transpose_b=True)
        grad = tf.matrix_transpose(grad)
        return grad

    def right_grad(U,S,V,dU,dV):
        US = tf.matmul(U,S)
        grad = tf.matmul(2*tf.matmul(US, msym(KT*(tf.matmul(V,dV,transpose_a=True))) ),V,transpose_b=True)
        return grad
    
    grad = tf.cond(tf.greater(v_sz, u_sz), lambda : left_grad(U,S,V,dU,dV), 
                                           lambda : right_grad(U,S,V,dU,dV))
    return [grad,None,None,None]

@function.Defun(tf.float32, tf.float32,tf.float32,tf.float32,func_name = 'EID', python_grad_func = gradient_eid)
def SVD_grad_map(x, s, u, v):
    return s,u,v 

