import numpy as np
import gym


def distance(mean_ini, mean_fin):
  bag = []
  for i in range(len(mean_ini)):
    bag.append((mean_ini[i] - mean_fin[i])**2)
  return np.sum(bag)**(0.5)

def norm(vector):
  return distance(vector, [0, 0])

def diff(vector1, vector2):
  dif = []
  for v, u in zip(vector1, vector2):
      dif.append([v[0] - u[0], v[1] - u[1]])
  return dif
    
def choose_data(num):
  num = num #random.randint(0,5)
  return 'DataIt0' + str(num) + '.xlsx'

#receive two means lists and return the second list with its means tidy
def tidy_obs(means1, means2):
  indices = []
  b_list = []
  for i in means1:
    dist = []
    for j in means2:
      dist.append(distance(i,j))  
    for j in means2:
      if np.argmin(dist) in b_list:
        dist[np.argmin(dist)] = 10000
        indices = indices
      else: 
        indices.append(np.argmin(dist))
        b_list.append(np.argmin(dist))
  return indices

def Gram_Schmidt_norm(matrix):

  U = matrix
  dim  = U.shape[0]
  E = np.zeros(matrix.shape)
  E[:,0] = U[:,0] 

  for j in range(1,dim):
    e = 0

    for i in range(j):

      e -= (np.dot(E[:,i],U[:,j])/np.dot(E[:,i],E[:,i])) * E[:,i]
      
    E[:,j] = U[:,j] + e
  for j in range(dim):
    E[:,j] = np.dot(E[:,j],E[:,j])**(-0.5) * E[:,j]

  return E
  
def spectral_composition(matrix, eigenvalues):
  matrix = Gram_Schmidt_norm(matrix)
  return np.matmul(np.matmul(matrix , np.diag(eigenvalues)), np.transpose(matrix))

def spectral_des_composition(matrix):
  A = matrix
  L, V = np.linalg.eig(A)
 
  return np.concatenate([np.concatenate(V), L])

