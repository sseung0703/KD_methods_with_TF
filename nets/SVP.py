import tensorflow as tf
from tensorflow.python.framework import function

def removenan(x):
    return tf.where(tf.is_finite(x), x,tf.zeros_like(x))

def SVD(X, n, name = None):
    with tf.variable_scope(name):
        sz = X.get_shape().as_list()
        if len(sz)>2:
            x = tf.reshape(X,[-1,sz[1]*sz[2],sz[3]])
            n = min(n, sz[1]*sz[2], sz[3])
        else:
            x = tf.expand_dims(X, 1)
            n = 1

        with tf.device('CPU'):
            g = tf.get_default_graph()
            with g.gradient_override_map({"Svd": "Svd_"}):
                s,u,v = tf.svd(x,full_matrices=False)
                
        s = removenan(s)
        v = removenan(v)
        u = removenan(u)
        
        s = tf.nn.l2_normalize(tf.slice(s,[0,0],[-1,n]),1)
        U = tf.nn.l2_normalize(tf.slice(u,[0,0,0],[-1,-1,n]),1)
        V = tf.nn.l2_normalize(tf.slice(v,[0,0,0],[-1,-1,n]),1)
        
        return s, U, V

def SVD_eid(X, n, name = None):
    with tf.variable_scope(name):
        sz = X.get_shape().as_list()
        if len(sz)>2:
            x = tf.reshape(X,[-1,sz[1]*sz[2],sz[3]])
            n = min(n, sz[1]*sz[2], sz[3])
        else:
            x = tf.expand_dims(X, 1)
            n = 1

        _, HW, D = x.get_shape().as_list()

        x_ = tf.stop_gradient(x)
        if HW/D < 3/2  and 2/3 < HW/D:
            with tf.device('CPU'):
                g = tf.get_default_graph()
                with g.gradient_override_map({"Svd": "Svd_"}):
                    s,u,v = tf.svd(x_,full_matrices=False)

        else:
            if HW < D:
                xxt = tf.matmul(x_,x_,transpose_b = True)
                with tf.device('CPU'):
                    _,u_svd,_ = tf.svd(xxt,full_matrices=False)
    
                v_svd = tf.matmul(x_, u_svd, transpose_a = True)
                s_svd = tf.linalg.norm(v_svd, axis = 1)
                v_svd = removenan(v_svd/tf.expand_dims(s_svd,1))
                
            else:
                xtx = tf.matmul(x_,x_,transpose_a = True)
                with tf.device('CPU'):
                    _, v_svd = tf.linalg.eigh(xtx)
                v_svd = tf.reshape(tf.image.flip_left_right(tf.reshape(v_svd,[-1,D,D,1])),[-1,D,D])
    
                u_svd = tf.matmul(x_, v_svd)
                s_svd = tf.linalg.norm(u_svd, axis = 1)
                u_svd = removenan(u_svd/tf.expand_dims(s_svd,1))
    
            s,u,v = SVD_grad_map(x,s_svd,u_svd,v_svd)
            s = tf.reshape(s,[-1,   min(HW,D)])
            u = tf.reshape(u,[-1,HW,min(HW,D)])
            v = tf.reshape(v,[-1, D,min(HW,D)])

        s = tf.nn.l2_normalize(tf.slice(s,[0,0],[-1,n])     ,1)
        U = tf.nn.l2_normalize(tf.slice(u,[0,0,0],[-1,-1,n]),1)
        V = tf.nn.l2_normalize(tf.slice(v,[0,0,0],[-1,-1,n]),1)
        
        return s, U, V
    
def Align_rsv(x, y):
    cosine = tf.matmul(x, y, transpose_a=True)
    mask = tf.where(tf.equal(tf.reduce_max(tf.abs(cosine), 2,keepdims=True), tf.abs(cosine)),
                    tf.sign(cosine), tf.zeros_like(cosine))
    x = tf.matmul(x, mask, transpose_b = True)
    return x, y

@tf.RegisterGradient('Svd_')
def gradient_svd(op, ds, dU, dV):
    s, U, V = op.outputs

    u_sz = tf.squeeze(tf.slice(tf.shape(dU),[1],[1]))
    v_sz = tf.squeeze(tf.slice(tf.shape(dV),[1],[1]))
    s_sz = tf.squeeze(tf.slice(tf.shape(ds),[1],[1]))

    S = tf.matrix_diag(s)
    s_2 = tf.square(s)

    eye = tf.expand_dims(tf.eye(s_sz),0) 
    k = (1 - eye)/(tf.expand_dims(s_2,2)-tf.expand_dims(s_2,1) + eye)
    KT = tf.matrix_transpose(k)
    KT = removenan(KT)
    
    def msym(X):
        return (X+tf.matrix_transpose(X))
    
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

def gradient_eid(op, ds, dU, dV):
    return gradient_svd(op, ds, dU, dV) + [None]*3

@function.Defun(tf.float32, tf.float32,tf.float32,tf.float32,func_name = 'EID', python_grad_func = gradient_eid)
def SVD_grad_map(x, s, u, v):
    return s,u,v 

