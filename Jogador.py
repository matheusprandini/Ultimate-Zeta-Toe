import random
from BuscaMinimax import BuscaMinimax 

class Jogador:
    
    def __init__(self, rede=None):
        self.rede = rede
    
    '''
    - Jogador Aleatório: executaJogadorAletorio -> Jogadas Aleatórias
    - Jogador MiniMax: executaJogadorMiniMax -> Jogadas baseadas no MiniMax
    '''
    
    #Função que executa um movimento aleatório dentre os movimentos legais disponíveis
    def executaJogadorAleatorio(self, ambiente, tabuleiro, marcacao):
        movimentos = list(ambiente.movimentosDisponiveisLegais(tabuleiro))    
        return random.choice(movimentos)
    
    def executaJogadorMiniMax(self, ambiente, tabuleiro, marcacao):
        minimax = BuscaMinimax(ambiente)
        resultado = minimax.algoritmoMinimax(ambiente, tabuleiro, marcacao, 4)
        movimento = resultado[1]
        return movimento
    
    def executaJogadorMiniMaxNN(self, ambiente, tabuleiro, marcacao):
        minimax = BuscaMinimax(ambiente, 'MLP', self.rede)
        resultado = minimax.algoritmoMinimax(ambiente, tabuleiro, marcacao, 100)
        movimento = resultado[1]
        return movimento