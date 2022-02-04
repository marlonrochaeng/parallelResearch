import numpy as np

class Utils():

  @staticmethod
  def initialize(path, jobs, resources):
    array = np.array(open(path).readlines(),dtype=float)
    ET = np.reshape(array,(jobs, resources))
    CT = ET.copy()
    maquinas = np.zeros(resources,dtype=float)
    return ET, CT, maquinas

  @staticmethod
  def get_min_in_matrix(matrix):
    '''
    Esta função retorna a linha e coluna do menor elemento da matriz passada por parametro
    '''
    return np.unravel_index(matrix.argmin(), matrix.shape)

  @staticmethod
  def get_max_in_matrix(matrix):
    '''
    Esta função retorna a linha e coluna do menor elemento da matriz passada por parametro
    '''
    return np.unravel_index(matrix.argmax(), matrix.shape)

  @staticmethod
  def get_min_in_array(array):
    '''
    Esta função a posição do menor elemento do array passado por parametro
    '''
    return np.where(array == array.min())[0][0]

  @staticmethod
  def get_max_in_array(array):
    '''
    Esta função a posição do menor elemento do array passado por parametro
    '''
    return np.where(array == array.max())[0][0]

  def minmin(self, ET,CT, maquinas):
    individuo = [np.inf for i in range(ET.shape[0])]
    pos = 0
    et_copy = ET.copy()
    while ET.shape[0] != 0:
      #print("et_shape", ET.shape)
      

      min_row, min_col = self.get_min_in_matrix(CT)
      '''
      print("min_row: ", min_row)
      print("min_col: ", min_col)
      print("menor elemento:", CT[min_row][min_col])
      print("CT[min_col]",CT[:5,min_col])
      print("------------------------\n")
      '''

      maquinas[min_col] += ET[min_row][min_col]

      for i in range(ET.shape[0]):
        CT[i][min_col] += ET[min_row][min_col]#maquinas[min_col]
      
      #print("CT[min_col]",CT[:5,min_col])

      #print("maquinas: ",maquinas)
      for i in range(len(et_copy)):
        if (ET[min_row] == et_copy[i]).all():
          pos = i
          break

      ET = np.delete(ET,(min_row),0)
      CT = np.delete(CT,(min_row),0)
      individuo[pos] = min_col

    return maquinas, individuo

  

  def maxmin2(self, ET,CT, maquinas):
    individuo = [np.inf for i in range(ET.shape[0])]
    pos = 0
    et_copy = ET.copy()
    while ET.shape[0] != 0:
      mins = []
      for i in range(ET.shape[0]):
        mins.append((i, self.get_min_in_array(CT[i]),CT[i][self.get_min_in_array(CT[i])]))
      
      #print("mins:", mins)
      #print("max of mins", max(mins, key=lambda item:item[1]))
      
      max_of_mins = max(mins, key=lambda item:item[2])
      min_exec = self.get_min_in_array(CT[max_of_mins[0]])

      maquinas[max_of_mins[1]] += ET[max_of_mins[0]][min_exec]

      for i in range(ET.shape[0]):
        CT[i][max_of_mins[1]] += ET[max_of_mins[0]][min_exec]

      for i in range(len(et_copy)):
        if (ET[max_of_mins[0]] == et_copy[i]).all():
          pos = i
          #print("Ind:", individuo)
          #print("Pos:", pos)
          #print("Maq:", min_exec)
          #input()
          break
      
      
        

      individuo[pos] = min_exec

      ET = np.delete(ET,(max_of_mins[0]),0)
      CT = np.delete(CT,(max_of_mins[0]),0)

    return maquinas, individuo


  def mct2(self, ET,CT, maquinas):
    #o correto
    individuo = [np.inf for i in range(ET.shape[0])]
    pos = 0
    
    while ET.shape[0] != 0:  
      temp = maquinas.copy()

      for i in range(len(temp)):
        temp[i] += ET[0][i]

      menor = self.get_min_in_array(temp)

      maquinas[menor] += ET[0][menor]
      individuo[pos] = menor
      pos += 1

      ET = np.delete(ET,(0),0)
    

    return maquinas, individuo

  def met(self, ET,CT, maquinas):
    individuo = [np.inf for i in range(ET.shape[0])]
    pos = 0

    while ET.shape[0] != 0:  

      menor = self.get_min_in_array(ET[0])

      maquinas[menor] += ET[0][menor]

      ET = np.delete(ET,(0),0)
      individuo[pos] = menor
      pos += 1

    return maquinas, individuo

  def olb(self, ET,CT, maquinas):
    individuo = [np.inf for i in range(ET.shape[0])]
    pos = 0

    while ET.shape[0] != 0:  
      temp = maquinas.copy()

      #for i in range(len(temp)):
      #  temp[i] += ET[0][i]

      menor = self.get_min_in_array(temp)

      maquinas[menor] += ET[0][menor]
      individuo[pos] = menor
      pos += 1

      ET = np.delete(ET,(0),0)

    return maquinas, individuo

  def sufferage(self, ET,CT, maquinas):
    ets = []
    sufferage = {}
    cts = []
    maquinas_cp = []

    while ET.shape[0] != 0:  
      maquinas_v = [False for i in range(len(maquinas))] 

      maquinas_op = {
        i: [] for i in range(len(maquinas))
      }
      #ultima daquela maquina, nao ultima escalonada
      for i in range(ET.shape[0]):
        
        ets.append(ET.copy())
        cts.append(CT.copy())
        maquinas_cp.append(maquinas.copy())

        temp = CT.copy()

        fst_minimum = self.get_min_in_array(temp[i])
        temp[i][fst_minimum] = np.inf
        scd_minimum = self.get_min_in_array(temp[i])

        #print("Value1: ", CT[i][fst_minimum])
        #print("Value2: ", CT[i][scd_minimum])

        sufferage[i] = CT[i][scd_minimum] - CT[i][fst_minimum]

        #sufferages.append(sufferage)
        print("Sufferage: ", sufferage)

        
        #input()

        if not maquinas_v[fst_minimum]:
          maquinas[fst_minimum] += ET[i][fst_minimum]
          maquinas_v[fst_minimum] = True

          maquinas_op[fst_minimum].append(i)

          for j in range(ET.shape[0]):
            CT[j][fst_minimum] += ET[i][fst_minimum]
          
          ET = np.delete(ET,(i),0)
          CT = np.delete(CT,(i),0)
          
        else:
          if sufferage[maquinas_op[fst_minimum][-1]] < sufferage[i]:
            
            ET = ets[-2].copy()
            CT = cts[-2].copy()

            i = i+1

            #fst_minimum = self.get_min_in_array(CT[i])
            maquinas = maquinas_cp[-1].copy()

            maquinas[fst_minimum] += ET[i][fst_minimum]
            for j in range(ET.shape[0]):
              CT[j][fst_minimum] += ET[i][fst_minimum]
            
            
            
            ET = np.delete(ET,(i),0)
            CT = np.delete(CT,(i),0)

          print("i:", i)
          print("Tamanho: ", ET.shape[0])
          print("Maquinas: ", maquinas)
           

      return maquinas