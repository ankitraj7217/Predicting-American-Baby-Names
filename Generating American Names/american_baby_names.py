import numpy as np
from utils import *
import random
import sys

sys.path.insert(0,"F:\CSE\Machine Learning\Deep Learning Assignment\Character level language model");
sys.path.insert(0,"F:\CSE\Machine Learning\Deep Learning Assignment\Character level language model\male_name.txt");

data = open('male_name.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))


char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
print(ix_to_char)


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
   
    for gradient in [dWax, dWaa, dWya, db, dby]:
        gradient=np.clip(gradient,-maxValue,maxValue,gradient)

    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients



def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
  
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
   
    x = np.zeros((vocab_size,1))
  
    a_prev = np.zeros((n_a,1))
   
    indices = []
    
    idx = -1 
    
    
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        
     
        a = np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,x)+b)
        z = np.dot(Wya,a)+by
        y = softmax(z)

        np.random.seed(counter+seed) 
        
        
        idx = np.random.choice(list(range(vocab_size)),p=y.ravel())

        indices.append(idx)
        
        x = np.zeros((vocab_size,1))
        x[idx] = 1 
        
        a_prev = a
        
   
        seed += 1
        counter +=1
    

    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices




def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    
   
    loss, cache = rnn_forward(X,Y,a_prev,parameters)
    
  
    gradients, a = rnn_backward(X,Y,parameters,cache)
    
    gradients = clip(gradients,5)
    
    parameters = update_parameters(parameters,gradients,learning_rate)
    

    
    return loss, gradients, a[len(X)-1]




def model(data, ix_to_char, char_to_ix, num_iterations = 100000, n_a = 100, dino_names = 8, vocab_size = 28):
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text, size of the vocabulary
    
    Returns:
    parameters -- learned parameters
    """
    n_x, n_y = vocab_size, vocab_size
    

    parameters = initialize_parameters(n_a, n_x, n_y)
    
    loss = get_initial_loss(vocab_size, dino_names)
    
    with open("male_name.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    
    np.random.seed(0)
    np.random.shuffle(examples)
    
    a_prev = np.zeros((n_a, 1))
 
    for j in range(num_iterations):
        
        index = j%len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]
        
        
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
        
        
        loss = smooth(loss, curr_loss)

       
        if j % 2000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                
                seed += 1  
      
            print('\n')
        
    return parameters


parameters = model(data, ix_to_char, char_to_ix)


def sample_with_char(parameters, char_to_ix, ind):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
  
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
   
    x = np.zeros((vocab_size,1))
    x[ind]=1
  
    a_prev = np.zeros((n_a,1))
   
    indices = []
    indices.append(ind)
    
    idx = -1 
    
    
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        
     
        a = np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,x)+b)
        z = np.dot(Wya,a)+by
        y = softmax(z)

        
        
        
        idx = np.random.choice(list(range(vocab_size)),p=y.ravel())

        indices.append(idx)
        
        x = np.zeros((vocab_size,1))
        x[idx] = 1 
        
        a_prev = a
        
   
       
        counter +=1
    

    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices


var=input("Enter first Character: \n")
sampled_indices = sample_with_char(parameters, char_to_ix,char_to_ix[var])
print_sample(sampled_indices, ix_to_char)
