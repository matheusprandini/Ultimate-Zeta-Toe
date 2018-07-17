import numpy as np
import itertools

class Utilidade:
    
    '''
    Atributos:
        
        - Ambiente: representa o tipo de jogo em que a busca minimax irá atuar;
    '''
    
    def __init__(self, ambiente, tipo='Estatica', rede=None):
        self.ambiente = ambiente
        self.tipo = tipo
        self.rede = rede
      
    #Função de Avaliação de um determinado ambiente 
    def funcaoAvaliacao(self, estado):
        
        '''
        Parâmetros:
            
            - Estado: representa o estado atual do tabuleiro do ambiente considerado;
            
        Retorno:
            
            - Pontuação para tal estado de entrada;
        '''
        
        if self.ambiente.getNome() == 'Jogo da Velha':
            if self.tipo == 'MLP':
                return self.utilidadeRedeNeuralMLP(estado)
            else:
                return self.funcaoAvaliacaoJogoDaVelha(estado)
        
    ##Função Utilidade que realiza a avaliação de uma determinado estado de tabuleiro, 
    #o qual tem dimensões n x n e com n marcações em sequência necessárias para vencer
    def funcaoAvaliacaoJogoDaVelha(self, estado):
        
        '''
        Parâmetros:
            
            - Estado: representa o estado atual do tabuleiro de um jogo da Velha;
            
        Retorno:
            
            - Pontuação para tal estado de entrada;
        '''
        
        resultado = 0
        
        for x in range(0,7,3):
            resultado += self.resultadoRetaJogoDaVelha([estado[x], estado[x+1], estado[x+2]])
        for x in range(3):
            resultado += self.resultadoRetaJogoDaVelha([estado[x], estado[x+3], estado[x+6]])
        
        #diagonais
        resultado += self.resultadoRetaJogoDaVelha([estado[0], estado[4], estado[8]])
        resultado += self.resultadoRetaJogoDaVelha([estado[2], estado[4], estado[6]])
    
        return resultado

    ##Função que retorna a avaliação de uma determinada reta
    def resultadoRetaJogoDaVelha(self, reta):
        
        '''
        Parâmetros:
            
            - Reta: representa uma linha, coluna ou diagonal do jogo;
            
        Retorno:
            
            - Pontuação com base nas marcações em tal reta;
        '''
        
        contadorMenosUm = reta.count(-1)
        contadorUm = reta.count(1)
        
        if contadorUm == 3 and contadorMenosUm == 0:
            return 1
        elif contadorMenosUm == 3 and contadorUm == 0:
            return -1
        '''elif contadorUm == 2 and contadorMenosUm == 0:
            return 0.2
        elif contadorMenosUm == 2 and contadorUm == 0:
            return -0.2'''
        return 0
    
    def utilidadeRedeNeuralMLP(self, estado):

        return self.rede.predicao(np.array([estado]))
        