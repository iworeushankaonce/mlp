# imports
import numpy as np
class MLP():
    '''
    This class implements a simple multilayer perceptron (MLP) with one hidden layer.
    The MLP is trained with true SGD.
    
    | **Args**
    | input_dim:             Number of input units.
    | hidden_layer_units:    Size of hidden layer.
    | eta:                   The learning rate.
    | initialization:        {'normal','xavier_normal','xavier_uniform'} 
    |                        weight initialization technique
    |                        'normal' by default
    | activation:            
    '''
    
    def __init__(self, input_dim=2, hidden_layer_units=10, eta=0.005,
                 initialization='normal',hidden_activation='tanh'):
        # input dimensions
        self.input_dim = input_dim
        # number of units in the hidden layer
        self.hidden_layer_units = hidden_layer_units
        # learning rate
        self.eta = eta
        # initialize weights 
        self.weights = list()
        if initialization == 'normal':
          self.weight_init_default()  
        elif initialization == 'xavier_normal':
          self.weight_init_xavier_normal()    
        elif initialization == 'xavier_uniform':
          self.weight_init_xavier_uniform()
        # set proper activation function
        self.a=list()
        self.a_d=list()
        if hidden_activation=='tanh':
          self.a.append(self.tanh)
          self.a_d.append(self.tanh_derivative)
        elif hidden_activation=='relu':
          self.a.append(self.relu)
          self.a_d.append(self.relu_derivative)
        
        self.a.append(self.tanh)
        self.a_d.append(self.tanh_derivative)             

    def tanh(self,a):
      return np.tanh(a)
    def tanh_derivative(self,a):
      return (1 - (self.tanh(a)**2))

    def relu(self,a):
      return np.clip(a, a_min=0, a_max=None)
    def relu_derivative(self,a):
      return  a > 0
     
    
    def l2_loss(self,a,b):
      return np.square(a-b).mean()
    def l2_loss_d(self):
      pass  

    def weight_init_default(self):
      self.weights.append(np.random.normal(0, 1, (self.hidden_layer_units, self.input_dim+1)))
      self.weights.append(np.random.normal(0, 1, (1, self.hidden_layer_units+1)))
    def weight_init_xavier_normal(self):
      weight_std=np.sqrt(2/(self.hidden_layer_units+self.input_dim+1))
      self.weights.append(np.random.normal(0, 1, (self.hidden_layer_units, self.input_dim+1)))
      weight_std=np.sqrt(2/(1 + self.hidden_layer_units+1))
      self.weights.append(np.random.normal(0, 1, (1, self.hidden_layer_units+1)))

    def weight_init_xavier_uniform(self):
      weight_range = np.sqrt(6/(self.hidden_layer_units+self.input_dim+1))
      self.weights.append(np.random.uniform(-weight_range, weight_range, 
                                            (self.hidden_layer_units, self.input_dim+1)))
      weight_range = np.sqrt(6/(1 + self.hidden_layer_units+1))
      self.weights.append(np.random.uniform(-weight_range, weight_range,
                                            (1, self.hidden_layer_units+1)))
      # initialization as described in equation 12 by Glorot & Bengio (2010)
            


    def predict(self,x,y,internal=False):
      bias = np.ones((1))
      X_b = np.concatenate((bias,x))
      x = np.reshape(X_b, (3,1))
      #forward pass
        
      a_l = np.matmul(self.weights[0], x)
      h_l = self.a[0](a_l); h_l = np.concatenate((np.ones((1,1)), h_l))

      a_m = np.matmul(self.weights[1],h_l)
      h_m = self.a[-1](a_m) 
      if internal:
        return (x,a_l,h_l,a_m,h_m)
      else:
        return h_m

    
    def evaluate(self,X,y):
      '''
      This method computes the mean squared error (MSE) 
      on the test set and returns the list of misclassified 
      points (using 0 as the classification threshold).
      | **Args**
      | X:                  Evaluation examples.
      | y:                  Ground truth labels.
      '''
      mse=list()
      misclassified_points=list()
      n_misclassified = 0

      for i,el in enumerate(X):
        y_pred = self.predict(X[i], y[i])
        mse.append((y_pred - y[i])**2)
        if(y_pred>=0 and y[i]==1) or (y_pred<0 and y[i]==-1):
          pass
        else:
          misclassified_points.append((i,el[0],el[1]))   
      return np.mean(mse),misclassified_points
    
    def train(self, X, y, epochs=100):
        '''
        Train function of the MLP class.
        This functions trains a MLP using true SGD with a constant learning rate.
        
        | **Args**
        | X:                  Training examples.
        | y:                  Ground truth labels.
        | epochs:             Number of epochs the MLP will be trained.
        '''

        # 1. compute activation h for a random data pair (forward pass)
        # 2. start with output layer, and traverse back by 5d

        loss=list()
        
        for e in range(epochs):
          shuffle_indxs = np.random.permutation(len(X))
          X = X[shuffle_indxs]
          y = y[shuffle_indxs]
          #if (e+1)%10==0:
          print("Epoch ",e+1)
          for i in range(len(y)):
            #forward pass
            (x,a_l,h_l,a_m,h_m) = self.predict(X[i],y[i],internal=True)
            
            error = (h_m-y[i])**2
            loss.append(error)
            #backward pass

            err_m = (self.a_d[-1](a_m)) * 2*(h_m-y[i])
            
            # the bias has to be removed from his function,
            # thus self.weights[1][:,1:].T
            err_l = (self.a_d[0](a_l)) * np.matmul(self.weights[1][:,1:].T, err_m)

            self.weights[1] -= self.eta*(np.outer(err_m, h_m))
            self.weights[0] -= self.eta*(np.outer(err_l, x)) 
          print("Loss ",np.around(np.mean(loss),8))
         

if __name__ == "__main__":
    '''
    Prepare dataset and train MLP 
    '''
    # load dataset
    dataset = np.load('xor.npy')
    # prepare training data and labels
    X, y = dataset[:,:2], dataset[:,2]
    # split into training and test
    X_train, X_test, y_train, y_test = X[:80000], X[80000:], y[:80000], y[80000:]