import tensorflow as tf
import numpy as np
import time
import pickle

class BSINN:
    # Initialize the class
    def __init__(self, X_1, X_test, u, u_test, Xf, Xf_test, uf, uf_test, 
                 Xb, Xb_test, ub, ub_test, Xi, Xi_test, ui, ui_test, 
                 u_layers, alpha):
        
        self.r = 0.02
        self.alpha = alpha
        self.batch_size = 32
        
        # total
        self.x = X_1[:,0:1]
        self.t = X_1[:,1:2]
        self.u = u
        self.v = X_1[:,2:3]
        
        self.x_test = X_test[:,0:1]
        self.t_test = X_test[:,1:2]       
        self.u_test = u_test
        self.v_test = X_test[:,2:3]
        
        #f
        self.xf = Xf[:,0:1]
        self.tf = Xf[:,1:2]
        self.uf = uf
        self.vf = Xf[:,2:3]
        
        self.xf_test = Xf_test[:,0:1]
        self.tf_test = Xf_test[:,1:2]       
        self.uf_test = uf_test
        self.vf_test = Xf_test[:,2:3] 
        
        # boundary
        self.xb = Xb[:,0:1]
        self.tb = Xb[:,1:2]       
        self.ub = ub
        self.vb = Xb[:,2:3]   
        
        self.xb_test = Xb_test[:,0:1]
        self.tb_test = Xb_test[:,1:2]       
        self.ub_test = ub_test
        self.vb_test = Xb_test[:,2:3] 
        
        # initial
        self.xi = Xi[:,0:1]
        self.ti = Xi[:,1:2]       
        self.ui = ui
        self.vi = Xi[:,2:3]
        
        self.xi_test = Xi_test[:,0:1]
        self.ti_test = Xi_test[:,1:2]       
        self.ui_test = ui_test
        self.vi_test = Xi_test[:,2:3]
        
        
        self.u_layers = u_layers
        #self.sigma_layers = sigma_layers
        
        # Initialize NNs
        self.u_weights, self.u_biases = self.initialize_NN(u_layers)
        #self.sigma_weights, self.sigma_biases = self.initialize_NN(sigma_layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.kappa = tf.constant([2.], dtype=tf.float32)  
        self.theta = tf.constant([0.055], dtype=tf.float32) 
        self.epsilon = tf.constant([2.], dtype=tf.float32)
        self.rho = tf.constant([-0.7], dtype=tf.float32)

        # Initialize parameters        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        
        self.xf_tf = tf.placeholder(tf.float32, shape=[None, self.xf.shape[1]])
        self.tf_tf = tf.placeholder(tf.float32, shape=[None, self.tf.shape[1]])
        self.uf_tf = tf.placeholder(tf.float32, shape=[None, self.uf.shape[1]])
        self.vf_tf = tf.placeholder(tf.float32, shape=[None, self.vf.shape[1]])
        
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.tb_tf = tf.placeholder(tf.float32, shape=[None, self.tb.shape[1]])
        self.ub_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        self.vb_tf = tf.placeholder(tf.float32, shape=[None, self.vb.shape[1]])
        
        self.xi_tf = tf.placeholder(tf.float32, shape=[None, self.xi.shape[1]])
        self.ti_tf = tf.placeholder(tf.float32, shape=[None, self.ti.shape[1]])
        self.ui_tf = tf.placeholder(tf.float32, shape=[None, self.ui.shape[1]])
        self.vi_tf = tf.placeholder(tf.float32, shape=[None, self.vi.shape[1]])
        
        self.lr_tf = tf.placeholder(tf.float32, shape=[])
        
        self.u_pred = self.net_u(self.x_tf, self.t_tf, self.v_tf)
        self.ub_pred = self.net_u(self.xb_tf, self.tb_tf, self.vb_tf)
        self.ui_pred = self.net_u(self.xi_tf, self.ti_tf, self.vi_tf)
        self.f_pred = self.net_f(self.xf_tf, self.tf_tf, self.vf_tf)
        
        
        self.loss = self.alpha*tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    (1-self.alpha)*(tf.reduce_mean(tf.square(self.f_pred)) + \
                                    tf.reduce_mean(tf.square(self.ub_tf - self.ub_pred)) + \
                                    tf.reduce_mean(tf.square(self.ui_tf - self.ui_pred)))
        
        #self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
        #                                                        method = 'L-BFGS-B', 
        #                                                        options = {'maxiter': 50000,
        #                                                                   'maxfun': 50000,
        #                                                                   'maxcor': 50,
        #                                                                   'maxls': 50,
        #
        #                                                                   'ftol' : 1.0 * np.finfo(float).eps})
        
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.lr_tf)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = (tf.add(tf.matmul(H, W), b))
        return Y
    
    def net_u(self, x, t, v):  
        u = self.neural_net(tf.concat([x,t,v],1), self.u_weights, self.u_biases)
        return u
    
    def net_f(self, x, t, v):
        kappa = self.kappa        
        theta = self.theta
        epsilon = self.epsilon
        rho = self.rho
        r = self.r
        u = self.net_u(x,t,v)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_v = tf.gradients(u, v)[0]
        u_vv = tf.gradients(u_v, v)[0]
        u_xv = tf.gradients(u_x, v)[0]

        f =  -u_t + (v/2)*x**2*u_xx + r*x*u_x + rho*epsilon*v**2*x*u_xv + (epsilon**2*v**3/2)*u_vv + v*(theta-kappa*v)*u_v - r*u
       
        return f
        
    #def callback(self, loss, kappa, theta, epsilon, rho):
    #    print('Loss: %.4e, kappa:%.3f, theta:%.3f, epsilon:%.3f, rho:%.3f'% (loss, kappa, theta, epsilon, rho))
        
    def train(self, nIter):
        
        start_time = time.time()
        
        step_number = int(np.ceil(max(self.x.shape[0], self.xi.shape[0], self.xb.shape[0])/self.batch_size))
        for it in range(nIter):

            for step in range(step_number):    
                
                offset = (step * self.batch_size) % (self.x.shape[0] - self.batch_size)
                offset_i = (step * self.batch_size) % (self.xi.shape[0] - self.batch_size)
                offset_b = (step * self.batch_size) % (self.xb.shape[0] - self.batch_size)
                # Generate a minibatch.
                
                batch_x = self.x[offset:(offset + self.batch_size)]
                batch_t = self.t[offset:(offset + self.batch_size)]
                batch_u = self.u[offset:(offset + self.batch_size)]
                batch_v = self.v[offset:(offset + self.batch_size)]
                
                batch_xi = self.xi[offset_i:(offset_i + self.batch_size)]
                batch_ti = self.ti[offset_i:(offset_i + self.batch_size)]
                batch_ui = self.ui[offset_i:(offset_i + self.batch_size)]
                batch_vi = self.vi[offset_i:(offset_i + self.batch_size)]
                
                batch_xb = self.xb[offset_b:(offset_b + self.batch_size)]
                batch_tb = self.tb[offset_b:(offset_b + self.batch_size)]
                batch_ub = self.ub[offset_b:(offset_b + self.batch_size)]
                batch_vb = self.vb[offset_b:(offset_b + self.batch_size)]
                
                tf_dict = {self.x_tf: batch_x, self.t_tf: batch_t, self.u_tf: batch_u, self.v_tf: batch_v,
                           self.xf_tf: batch_x, self.tf_tf: batch_t, self.uf_tf: batch_u, self.vf_tf: batch_v,
                           self.xb_tf: batch_xb, self.tb_tf: batch_tb, self.ub_tf: batch_ub, self.vb_tf: batch_vb,
                           self.xi_tf: batch_xi, self.ti_tf: batch_ti, self.ui_tf: batch_ui, self.vi_tf: batch_vi,
                          self.lr_tf: 0.001}
                    
                    
                if it >= 0 and it <= 2000:
                    tf_dict.update({self.lr_tf: 0.001})
                if it > 2000 and it <= 3500:
                    tf_dict.update({self.lr_tf: 0.0005})
                if it > 3500 and it <= 5000:
                    tf_dict.update({self.lr_tf: 0.00001})
                    
                 
                self.sess.run(self.train_op_Adam, tf_dict)
                     
                
                # Print
                #if it % 1000 == 0:
                #    elapsed = time.time() - start_time
                #    loss_value = self.sess.run(self.loss, tf_dict)
                #    print('It: %d, Loss: %.3e, Time: %.2f' % 
                #          (it, loss_value, elapsed))
                #    start_time = time.time()
        
        
    def predict_all(self, X):
        
        tf_dict = {self.x_tf: X[:,0:1], self.t_tf: X[:,1:2], self.v_tf: X[:,2:3]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        #f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star#, f_star

    def predict_test(self, X_test):
        
        tf_dict = {self.x_tf: X_test[:,0:1], self.t_tf: X_test[:,1:2], self.v_tf: X_test[:,2:3]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        #f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star#, f_star
    
    
    
if __name__ == "__main__": 
    
    # Train/test data
    infile = open('Three_half_4_folds_data_v_5000','rb')
    data = pickle.load(infile)
    infile.close()
    
    Data_test = data['Data_test']
    Data_train = data['Data_train']
    bd_train = data['bd_train']
    bd_test = data['bd_test']
    ini_train = data['ini_train']
    ini_test = data['ini_test']
    
    Data_test_tot = []
    for i in range(4):
        Data_test_tot.append(np.concatenate((Data_test[i],bd_test[i],ini_test[i])))
    
    Data_train_tot = []
    for i in range(4):
        Data_train_tot.append(np.concatenate((Data_train[i],bd_train[i],ini_train[i])))
    
    # Pseud-train/test data
    infile = open('Three_half_4_folds_data_s_v_5000','rb')
    data = pickle.load(infile)
    infile.close()
    
    Data_test_s = data['Data_test_s']
    Data_train_s = data['Data_train_s']
    bd_train_s = data['bd_train_s']
    bd_test_s = data['bd_test_s']
    ini_train_s = data['ini_train_s']
    ini_test_s = data['ini_test_s']
    
    
    Data_test_tot_s = []
    for i in range(4):
        Data_test_tot_s_i = []
        for j in range(4):
            Data_test_tot_s_i.append(np.concatenate((Data_test_s[i][j],bd_test_s[i][j],ini_test_s[i][j])))
        Data_test_tot_s.append(Data_test_tot_s_i)
    
    
    num_layers = [4]
    num_neurons = [20]
    seed_num = [1234,2341,3412,4123]
    
    for l in num_layers:
        for n in num_neurons:
            u_layers = np.concatenate([[3], n*np.ones(l), [1]]).astype(int).tolist()
            #sigma_layers = np.concatenate([[1], n*np.ones(l), [1]]).astype(int).tolist()
            
            
            ###### k-fold cross-validation#####
            k = 4      #number of folds
            
            c=0
            for alpha in np.linspace(0,1,5):
                c+=1
                
                
                Total_error_u_num = 0
                Total_error_u_den = 0
                Total_error_u_test_num = 0
                Total_error_u_test_den = 0
                Total_error_u_train_num = 0
                Total_error_u_train_den = 0
                
                for i in range(k):
                    
                    # total for u
                    X_u_test_1 = Data_test[i][:,0:3]
                    u_test = Data_test[i][:,3][:,None]
                    K_test = Data_test[i][:,4][:,None]
                    
                    X_u_train_1 = Data_train[i][:,0:3]
                    u_train = Data_train[i][:,3][:,None]
                    K_train = Data_train[i][:,4][:,None]
                    
                    # domain
                    X_f_test_1 = Data_test[i][:,0:3]
                    uf_test = Data_test[i][:,3][:,None]
                    Kf_test = Data_test[i][:,4][:,None]
                    
                    X_f_train_1 = Data_train[i][:,0:3]
                    uf_train = Data_train[i][:,3][:,None]
                    Kf_train = Data_train[i][:,4][:,None]
                    
                    # boundary
                    X_b_test_1 = bd_test[i][:,0:3]
                    ub_test = bd_test[i][:,3][:,None]
                    Kb_test = bd_test[i][:,4][:,None]
                    
                    X_b_train_1 = bd_train[i][:,0:3]
                    ub_train = bd_train[i][:,3][:,None]
                    Kb_train = bd_train[i][:,4][:,None]
                    
                    # initial
                    X_i_test_1 = ini_test[i][:,0:3]
                    ui_test = ini_test[i][:,3][:,None]
                    Ki_test = ini_test[i][:,4][:,None]
                    
                    X_i_train_1 = ini_train[i][:,0:3]
                    ui_train = ini_train[i][:,3][:,None]
                    Ki_train = ini_train[i][:,4][:,None]                
                    
                    X_test_tot = Data_test_tot[i][:,0:3]
                    u_test_tot = Data_test_tot[i][:,3][:,None]
                    K_test_tot = Data_test_tot[i][:,4][:,None]
                    
                    X_train_tot = Data_train_tot[i][:,0:3]
                    u_train_tot = Data_train_tot[i][:,3][:,None]
                    K_train_tot = Data_train_tot[i][:,4][:,None]

                    

                    
                    tf.reset_default_graph()
                    np.random.seed(seed_num[i])
                    tf.set_random_seed(seed_num[i])
                    
                    model = BSINN(X_u_train_1, X_u_test_1, u_train, u_test, 
                                              X_f_train_1, X_f_test_1, uf_train, uf_test, 
                                              X_b_train_1, X_b_test_1, ub_train, ub_test, 
                                              X_i_train_1, X_i_test_1, ui_train, ui_test, 
                                              u_layers, alpha)
                
                    model.train(5000)
                    
                    X_star_1 = np.concatenate([X_train_tot,X_test_tot])
                    u_star = np.concatenate([u_train_tot,u_test_tot])
                    K = np.concatenate([K_train_tot,K_test_tot])
                    
                    u_pred = model.predict_all(X_star_1)
                    
                    error_u_num = (np.linalg.norm(u_star*K-u_pred*K,2))**2
                    error_u_den = (np.linalg.norm(u_star*K,2))**2
                    
                    #############train error#############
                    u_train_pred = model.predict_test(X_train_tot)
                    # Truncate negative option price
                    u_min = np.maximum((X_train_tot[:,0:1]-np.exp(-model.r*X_train_tot[:,1:2])),0)
                    
                    for q in range(len(u_train_pred)):
                        if u_train_pred[q] < u_min[q]:
                            u_train_pred[q] = u_min[q] # actually no need loop, just use maximum function
                            
                    error_u_train_num = (np.linalg.norm(u_train_tot*K_train_tot-u_train_pred*K_train_tot))**2
                    error_u_train_den = (np.linalg.norm(u_train_tot*K_train_tot))**2
                    ####################################
                    
                    
                    #############test error#############
                    u_test_pred = model.predict_test(X_test_tot)
                    # Truncate negative option price
                    u_min = np.maximum((X_test_tot[:,0:1]-np.exp(-model.r*X_test_tot[:,1:2])),0)
                    
                    for q in range(len(u_test_pred)):
                        if u_test_pred[q] < u_min[q]:
                            u_test_pred[q] = u_min[q]
                            
                    error_u_test_num = (np.linalg.norm(u_test_tot*K_test_tot-u_test_pred*K_test_tot))**2
                    error_u_test_den = (np.linalg.norm(u_test_tot*K_test_tot))**2
                    ####################################
            
                    Total_error_u_num += error_u_num
                    Total_error_u_den += error_u_den
                    Total_error_u_test_num += error_u_test_num
                    Total_error_u_test_den += error_u_test_den
                    Total_error_u_train_num += error_u_train_num
                    Total_error_u_train_den += error_u_train_den
                    
                    #print('#################################################################')
                    #print('Current error for big fold_%d with alpha %.3f : %.4e'%(i+1,alpha,np.sqrt(error_u_train_num/error_u_train_den)))
                    #print('Cumulative error big fold_%d: %.4e'%(i+1,np.sqrt(Total_error_u_train_num/Total_error_u_train_den)))
                    #print('Current error for big fold_%d with alpha %.3f : %.4e'%(i+1,alpha,np.sqrt(error_u_test_num/error_u_test_den)))
                    #print('Cumulative error big fold_%d: %.4e'%(i+1,np.sqrt(Total_error_u_test_num/Total_error_u_test_den)))
                    
                    
                    Three_half_data = {'X_u_test_1':X_u_test_1, 'X_star_1': X_star_1, 'X_test_tot':X_test_tot, 'u_test_pred':u_test_pred, 
                                  'u_test_tot':u_test_tot, 'K_test_tot':K_test_tot,'alpha':alpha,
                                 'X_train_tot':X_train_tot, 'u_train_pred':u_train_pred, 'u_train_tot':u_train_tot,
                                       'K_train_tot':K_train_tot}
                    filename = 'Three_half_%d_l_%d_n_%d_constant_fitsolver_alpha_%d_rep_v(with_train)'%(l,n,i+1,c)
                    outfile = open(filename,'wb')
                    pickle.dump(Three_half_data,outfile)
                    outfile.close()
                
                #print('architecture: %dl%dn%dfold ##################################################'%(l,n,i+1))
                #print('Total Error u: %.4e' % (np.sqrt(Total_error_u_num/Total_error_u_den)))
                #print('Total Error u_test: %.4e' % (np.sqrt(Total_error_u_test_num/Total_error_u_test_den)))
                
            
 