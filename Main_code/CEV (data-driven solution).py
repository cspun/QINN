import tensorflow as tf
import numpy as np
import time
import pickle
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()


class BSINN:
    # Initialize the class
    def __init__(self, X_1, X_test, u, u_test, 
                 Xf, Xf_test, uf, uf_test, Kf, 
                 Xb, Xb_test, ub, ub_test, 
                 Xi, Xi_test, ui, ui_test, 
                 u_layers, alpha):
        
        self.r = 0.02
        self.alpha = alpha
        self.batch_size = 32

        # total
        self.x = X_1[:,0:1]
        self.t = X_1[:,1:2]
        self.u = u
        
        self.x_test = X_test[:,0:1]
        self.t_test = X_test[:,1:2]       
        self.u_test = u_test
        
        #f
        self.xf = Xf[:,0:1]
        self.tf = Xf[:,1:2]
        self.uf = uf
        self.Kf = Kf
        
        self.xf_test = Xf_test[:,0:1]
        self.tf_test = Xf_test[:,1:2]       
        self.uf_test = uf_test
        
        # boundary
        self.xb = Xb[:,0:1]
        self.tb = Xb[:,1:2]       
        self.ub = ub
        
        self.xb_test = Xb_test[:,0:1]
        self.tb_test = Xb_test[:,1:2]       
        self.ub_test = ub_test

        # initial
        self.xi = Xi[:,0:1]
        self.ti = Xi[:,1:2]       
        self.ui = ui
        
        self.xi_test = Xi_test[:,0:1]
        self.ti_test = Xi_test[:,1:2]       
        self.ui_test = ui_test
        
        
        
        self.u_layers = u_layers
        #self.sigma_layers = sigma_layers
        
        # Initialize NNs
        self.u_weights, self.u_biases = self.initialize_NN(u_layers)
        #self.sigma_weights, self.sigma_biases = self.initialize_NN(sigma_layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        
        #self.sigma = tf.Variable([2.], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0., 3.)) #1.83 ini=2
        #self.gamma = tf.Variable([1.], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0., 2.)) #0.8 ini=1
        
        self.sigma = tf.constant([1.83]) 
        self.gamma = tf.constant([0.8])
        
        # Initialize parameters        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.xf_tf = tf.placeholder(tf.float32, shape=[None, self.xf.shape[1]])
        self.tf_tf = tf.placeholder(tf.float32, shape=[None, self.tf.shape[1]])
        self.uf_tf = tf.placeholder(tf.float32, shape=[None, self.uf.shape[1]])
        self.Kf_tf = tf.placeholder(tf.float32, shape=[None, self.Kf.shape[1]])
        
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.tb_tf = tf.placeholder(tf.float32, shape=[None, self.tb.shape[1]])
        self.ub_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        
        self.xi_tf = tf.placeholder(tf.float32, shape=[None, self.xi.shape[1]])
        self.ti_tf = tf.placeholder(tf.float32, shape=[None, self.ti.shape[1]])
        self.ui_tf = tf.placeholder(tf.float32, shape=[None, self.ui.shape[1]])
        
        self.lr_tf = tf.placeholder(tf.float32, shape=[])
        
        self.u_pred = self.net_u(self.x_tf, self.t_tf)
        self.ub_pred = self.net_u(self.xb_tf, self.tb_tf)
        self.ui_pred = self.net_u(self.xi_tf, self.ti_tf)
        self.f_pred = self.net_f(self.xf_tf, self.tf_tf, self.Kf_tf)

        
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
    
    
    def net_u(self, x, t):  
        u = self.neural_net(tf.concat([x,t],1), self.u_weights, self.u_biases)
        return u
    
    def net_f(self, x, t, K):
        
        r = self.r
        u = self.net_u(x,t)
        gamma = self.gamma
        sigma = self.sigma 
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = -u_t + 0.5*sigma**2*K**(2*gamma-2)*x**(2*gamma)*u_xx + r*x*u_x - r*u
        
        return f
        
    #def callback(self, loss, sigma, gamma):
    #    print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, sigma, gamma))
    
    def get_batches(self, data, batch_size):
    
        batch_data = []
        
        for i in range(int(data.shape[0]/batch_size)):
            
            if i== int(data.shape[0]/batch_size) - 1:
                data_batch = data[i*batch_size ::, :]
            else:
                data_batch = data[i*batch_size : (i+1)*batch_size, :]
    
            batch_data.append(data_batch)
    
        return batch_data
    
    def train(self, nIter):
        
        start_time = time.time()
        
        batch_x = self.get_batches(self.x, self.batch_size) 
        batch_t = self.get_batches(self.t, self.batch_size) 
        batch_u = self.get_batches(self.u, self.batch_size)
        batch_K = self.get_batches(self.Kf, self.batch_size)
        
        batch_xb = self.get_batches(self.xb, self.batch_size) 
        batch_tb = self.get_batches(self.tb, self.batch_size) 
        batch_ub = self.get_batches(self.ub, self.batch_size)  
        
        batch_xi = self.get_batches(self.xi, self.batch_size) 
        batch_ti = self.get_batches(self.ti, self.batch_size) 
        batch_ui = self.get_batches(self.ui, self.batch_size) 
        
        
        for it in range(nIter):
            
            for p in range(len(batch_x)):
                tf_dict = {self.x_tf: batch_x[p], self.t_tf: batch_t[p], self.u_tf: batch_u[p],
                           self.xf_tf: batch_x[p], self.tf_tf: batch_t[p], self.uf_tf: batch_u[p], self.Kf_tf: batch_K[p],
                           self.xb_tf: batch_xb[p], self.tb_tf: batch_tb[p], self.ub_tf: batch_ub[p],
                           self.xi_tf: batch_xi[p], self.ti_tf: batch_ti[p], self.ui_tf: batch_ui[p],
                          self.lr_tf: 0.001}
                
                if it >= 0 and it <= 2500:
                    tf_dict.update({self.lr_tf: 0.001})
                    
                if it > 2500 and it <= 5000:
                    tf_dict.update({self.lr_tf: 0.0005})
                    
                if it > 5000 and it <= 7500:
                    tf_dict.update({self.lr_tf: 0.00001})
                    
                if it > 7500 and it <= 10000:
                    tf_dict.update({self.lr_tf: 0.000001})
                    
                self.sess.run(self.train_op_Adam, tf_dict)
                
                #lr rate for variable
                #if it > 0 and it <= 10000:
                #    tf_dict.update({self.lr_tf: 0.001})
                #    
                #if it > 10000 and it <= 15000:
                #    tf_dict.update({self.lr_tf: 0.0005})
                #    
                #if it > 15000 and it <= 20000:
                #    tf_dict.update({self.lr_tf: 0.000001})                    
                #self.sess.run(self.train_op_Adam, tf_dict)
                
           
                # Print
                #if it % 1000 == 0:
                #    elapsed = time.time() - start_time
                #    loss_value = self.sess.run(self.loss, tf_dict)
                #    sigma_value = self.sess.run(self.sigma, tf_dict)
                #    gamma_value = self.sess.run(self.gamma, tf_dict)
                #    print('It: %d, Loss: %.3e, sigma: %.3f, gamma: %.3f, Time: %.2f' % 
                #          (it, loss_value, sigma_value, gamma_value, elapsed))
                #    start_time = time.time()

        
        
    def predict_all(self, X):
        
        tf_dict = {self.x_tf: X[:,0:1], self.t_tf: X[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        #f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star#, f_star

    def predict_test(self, X_test):
        
        tf_dict = {self.x_tf: X_test[:,0:1], self.t_tf: X_test[:,1:2]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        #f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star#, f_star
    
    
    
if __name__ == "__main__": 
    
    # Train/test data
    infile = open('CEV_4_folds_data','rb')
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
    infile = open('CEV_4_folds_data_s','rb')
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
        
    
    num_layers = [3]
    num_neurons = [20]
    seed_num = [1234,2341,3412,4123]
    
    for l in num_layers:
        for n in num_neurons:
            u_layers = np.concatenate([[2], n*np.ones(l), [1]]).astype(int).tolist()
            #sigma_layers = np.concatenate([[1], n*np.ones(l), [1]]).astype(int).tolist()


            ###### k-fold cross-validation#####
            k = 4      #number of folds                
                
            c=0
            for alpha in np.linspace(0.,1.,5):
                c += 1
                Total_error_u_num = 0
                Total_error_u_den = 0
                Total_error_u_test_num = 0
                Total_error_u_test_den = 0     
                Total_error_u_train_num = 0
                Total_error_u_train_den = 0

                for i in range(k):
                    
                    tf.reset_default_graph()
                    np.random.seed(seed_num[i])
                    tf.set_random_seed(seed_num[i])
                    
                    # total for u
                    X_u_test_1 = Data_test[i][:,0:2]
                    u_test = Data_test[i][:,2][:,None]
                    K_test = Data_test[i][:,3][:,None]
                    
                    X_u_train_1 = Data_train[i][:,0:2]
                    u_train = Data_train[i][:,2][:,None]
                    K_train = Data_train[i][:,3][:,None]
                    
                    # domain
                    X_f_test_1 = Data_test[i][:,0:2]
                    uf_test = Data_test[i][:,2][:,None]
                    Kf_test = Data_test[i][:,3][:,None]
                    
                    X_f_train_1 = Data_train[i][:,0:2]
                    uf_train = Data_train[i][:,2][:,None]
                    Kf_train = Data_train[i][:,3][:,None]
                    
                    # boundary
                    X_b_test_1 = bd_test[i][:,0:2]
                    ub_test = bd_test[i][:,2][:,None]
                    Kb_test = bd_test[i][:,3][:,None]
                    
                    X_b_train_1 = bd_train[i][:,0:2]
                    ub_train = bd_train[i][:,2][:,None]
                    Kb_train = bd_train[i][:,3][:,None]
                    
                    # initial
                    X_i_test_1 = ini_test[i][:,0:2]
                    ui_test = ini_test[i][:,2][:,None]
                    Ki_test = ini_test[i][:,3][:,None]
                    
                    X_i_train_1 = ini_train[i][:,0:2]
                    ui_train = ini_train[i][:,2][:,None]
                    Ki_train = ini_train[i][:,3][:,None]                
                    
                    X_test_tot = Data_test_tot[i][:,0:2]
                    u_test_tot = Data_test_tot[i][:,2][:,None]
                    K_test_tot = Data_test_tot[i][:,3][:,None]
                    
                    X_train_tot = Data_train_tot[i][:,0:2]
                    u_train_tot = Data_train_tot[i][:,2][:,None]
                    K_train_tot = Data_train_tot[i][:,3][:,None]

                  
                    
                    model = BSINN(X_u_train_1, X_u_test_1, u_train, u_test, 
                                              X_f_train_1, X_f_test_1, uf_train, uf_test, Kf_train, 
                                              X_b_train_1, X_b_test_1, ub_train, ub_test, 
                                              X_i_train_1, X_i_test_1, ui_train, ui_test, 
                                              u_layers, alpha)
                    
                    model.train(10000)
                    
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
                    #print('Current error for big fold_%d: %.4e'%(i+1,np.sqrt(error_u_test_num/error_u_test_den)))
                    #print('Cumulative error big fold_%d: %.4e'%(i+1,np.sqrt(Total_error_u_test_num/Total_error_u_test_den)))
                    #
                    sigma = model.sess.run(model.sigma)
                    gamma = model.sess.run(model.gamma)
                    #print('sigma: %f'%sigma)
                    #print('gamma: %f'%gamma)
                   
                    
                    CEV_data = {'X_u_test_1':X_u_test_1, 'X_star_1': X_star_1, 'X_test_tot':X_test_tot, 'u_test_pred':u_test_pred, 
                                  'u_test_tot':u_test_tot, 'K_test_tot':K_test_tot,'alpha':alpha,
                                 'X_train_tot':X_train_tot, 'u_train_pred':u_train_pred, 'u_train_tot':u_train_tot,
                                'K_train_tot':K_train_tot}
                    filename = 'CEV_%d_l_%d_n_%d_fold_constant_fitsolver_alpha_%d_rep_v(with_train)'%(l,n,i+1,3)
                    outfile = open(filename,'wb')
                    pickle.dump(CEV_data,outfile)
                    outfile.close()
            
                #print('architecture: %dl%dn%dfold ##################################################'%(l,n,i+1))
                #print('Total Error u: %.4e' % (np.sqrt(Total_error_u_num/Total_error_u_den)))
                #print('Total Error u_test: %.4e' % (np.sqrt(Total_error_u_test_num/Total_error_u_test_den)))
                