import numpy as np
import math
from sklearn.preprocessing import normalize

class RedeNeuralMLP:
    
    def __init__(self, numNeuroniosEntrada=None, numNeuroniosOculta=None, numNeuroniosSaida=None, parametros={}):
        
        self.numNeuroniosCamadaEntrada = numNeuroniosEntrada
        self.numNeuroniosCamadaOculta = numNeuroniosOculta
        self.numNeuroniosCamadaSaida = numNeuroniosSaida
        self.parametros = parametros
    
    #Rede Neural com apenas 1 camada oculta
    def inicializarParametros2Camadas(self):
        
        """
        Returno:
        
            - parametros --dicionario contendo os seguintes parâmetro:
                        W1 -- matriz de pesos de dimensão (numNeuroniosOculta, numNeuroniosEntrada)
                        b1 -- vetor de bias de dimensão (numNeuroniosEntrada, 1)
                        W2 -- matriz de pesos de dimensão (numNeuroniosSaida, numNeuroniosOculta)
                        b2 -- vetor de bias de dimensão (numNeuroniosSaida, 1)
        """
        
        #np.random.seed(2)
        
        W1 = np.random.uniform(-1,1,(self.numNeuroniosCamadaOculta, self.numNeuroniosCamadaEntrada))
        b1 = np.zeros(shape=(self.numNeuroniosCamadaOculta, 1))
        W2 = np.random.uniform(-1,1,(self.numNeuroniosCamadaSaida, self.numNeuroniosCamadaOculta))
        b2 = np.zeros(shape=(self.numNeuroniosCamadaSaida, 1))

        assert(W1.shape == (self.numNeuroniosCamadaOculta, self.numNeuroniosCamadaEntrada))
        assert(b1.shape == (self.numNeuroniosCamadaOculta, 1))
        assert(W2.shape == (self.numNeuroniosCamadaSaida, self.numNeuroniosCamadaOculta))
        assert(b2.shape == (self.numNeuroniosCamadaSaida, 1))
        
        parametros = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
        
        self.parametros = parametros
    
    def propagacaoParaFrente(self, X):
        """
        Argumentos:
        X -- dados de entrada (numNeuroniosEntrada, array de entrada)
        parametros -- dicionário contendo os dados dos pesos e bias
        
        Retorno:
        A2 -- A saída da segunda ativação (tangente hiperbólica)
        cache -- dicionário contendo os parâmetros "Z1", "A1", "Z2" e "A2"
        """
        
        # Recupera cada parâmetro do dicionário "parametros"
        W1 = self.parametros['W1']
        b1 = self.parametros['b1']
        W2 = self.parametros['W2']
        b2 = self.parametros['b2']

        # Implementa o forward propagation
        Z1 = np.dot(W1, X.T) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = np.tanh(Z2)
        #A2 = self.sigmoid(Z2) #sigmoid
        
        assert(A2.shape == (1, X.shape[0]))
        
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
        
        return A2, cache
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def PropagacaoParaTras(self, cache, X, Y):
        """      
        Argumentos:
        parametros -- dicionário contendo parametros
        cache -- dicionário  contendo os parâmetros "Z1", "A1", "Z2" e "A2".
        X -- dado de entrada
        Y -- saída desejada (true label)
        
        Retorno:
        gradientes -- dicionário contendo os gradientes
        """
        
        # Recupera W1 e W2 do dicionário "parametros".
        W1 = self.parametros['W1']
        W2 = self.parametros['W2']
            
        #Recupera A1 e A2 do dicionário "cache".
        A1 = cache['A1']
        A2 = cache['A2']
        
        m = X.shape[1]

        # Backward propagation: calcula dW1, db1, dW2, db2. 
        dZ2 = A2 - Y.T
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        gradientes = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        
        return gradientes
    
    def atualizarParametros(self, gradientes, taxaAprendizagem=0.1, momentum=1):
        """    
        Argumentos:
        parametros -- dicionário contendo parâmetros
        gradientes -- dicionário contendo os gradientes 
        
        """
        # Recupera cada parâmetro do dicionário "parametros"
        W1 = self.parametros['W1']
        b1 = self.parametros['b1']
        W2 = self.parametros['W2']
        b2 = self.parametros['b2']
        
        # Recupera cada gradiente do dicionário "gradientes"
        dW1 = gradientes['dW1']
        db1 = gradientes['db1']
        dW2 = gradientes['dW2']
        db2 = gradientes['db2']
        
        # Atualização de cada parâmetro
        W1 = (W1 * momentum) - (taxaAprendizagem * dW1)
        b1 = (b1 * momentum) - (taxaAprendizagem * db1)
        W2 = (W2 * momentum) - (taxaAprendizagem * dW2)
        b2 = (b2 * momentum) - (taxaAprendizagem * db2)

        parametros = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
        
        self.parametros = parametros
        
    def calcularCusto(self, A2, Y):
        """
        Calcula o custo de entropia cruzada
        
        Argumentos:
        A2 -- Saídas obtidas da rede
        Y -- Saídas desejadas
        
        Retorno:
        custo -- custo de entropia cruzada
        """
        Y = Y.T
        
        m = Y.shape[1] #número de exemplos
        epsilon = 1e-15
        
        
        #Valores menores que epsilon assumirão valor epsilon;
        #Valores maiores que 0.9999999 assumirão valor 0.9999999
        A2_clipped = np.clip(A2, epsilon, 0.9999999)
        
        # Cálculo do custo de entropia cruzada
        logprobs = np.multiply(np.log(np.abs(A2_clipped)), Y) + np.multiply((1 - Y), np.log(np.abs(1 - A2_clipped)))
        custo = np.multiply(-1, np.sum(logprobs) / m)
        
        custo = np.squeeze(custo) #Garante que a dimensão esteja correta
    
        assert(isinstance(custo, float))
        
        return custo
    
    def executaTreinamentoModelo2Camadas(self, X, Y, numIteracoes=100000, taxaAprendizagem=0.1, momentum=1):
        """
        Argumentos:
        X -- entrada
        Y -- saída desejada
        numIteracoes -- número de iterações
        
        """
        #numeroNeuroniosCamadaEntrada = self.inicializarTamanhoCamadas(X, Y)[0]
        #numeroNeuroniosCamadaOculta = self.inicializarTamanhoCamadas(X, Y)[1]
        #numeroNeuroniosCamadaSaida = self.inicializarTamanhoCamadas(X, Y)[2]
        
        custos = []
        iteracoes = []
        
        # Inicializa parâmetros
        self.inicializarParametros2Camadas()
        
        #Normalização da entrada
        #X = normalize(X)
        
        # Loop (gradiente)
        for i in range(0, numIteracoes):
             
            # Forward propagation
            A2, cache = self.propagacaoParaFrente(X)
            
            # Função de Custo.
            custo = self.calcularCusto(A2, Y)
     
            # Backpropagation
            gradientes = self.PropagacaoParaTras(cache, X, Y)
     
            # Atualização dos parâmetros
            self.atualizarParametros(gradientes, taxaAprendizagem, momentum)
            
            # Exibe custo a cada 1000 iterações
            if i % (numIteracoes / 10) == 0:
                custos.append(custo)
                iteracoes.append(i)
                print ("Custo na iteração %i: %f" % (i, custo))
        
        print("Custo Final: ", custo)
        custos.append(custo)
        iteracoes.append(numIteracoes)
        
        return custos, iteracoes
    
    def executaTreinamentoEntrada(self, X, Y, taxaAprendizagem=0.1, momentum=1):
        """
        Argumentos:
        X -- entrada
        Y -- saída desejada
        taxaAprendizagem -- alfa
        momentum -- momentum
        """

        # Forward propagation
        A2, cache = self.propagacaoParaFrente(X)
        
        # Backpropagation
        gradientes = self.PropagacaoParaTras(cache, X, Y)
 
        # Atualização dos parâmetros
        self.atualizarParametros(gradientes, taxaAprendizagem, momentum)
    
    def predicao(self, X):
        """        
        Argumentos:
        parametros -- dicionário que contém os parâmetros
        X -- entrada
        
        Retorno
        predição -- saída da entrada X dada pela rede MLP
        """
        
        A2, cache = self.propagacaoParaFrente(X)
        predicao = A2
        
        return predicao
    
        