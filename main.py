from AmbienteTicTacToe import AmbienteTicTacToe
from Jogador import Jogador
from Partida import Partida
from RedeNeuralMLP import RedeNeuralMLP
from TreinamentoSupervisionado import TreinamentoSupervisionado
import numpy as np
import timeit

def campeonatoTeste():
    
    jogador1 = Jogador()
    jogador2 = Jogador()
    
    contEmpate = 0
    cont1 = 0
    cont2 = 0
    
    for i in range(0,100):
        
        ambiente = AmbienteTicTacToe(3)
        
        partida = Partida(ambiente)
        
        result = partida.partidaJogoDaVelha(jogador1.executaJogadorAleatorio, jogador2.executaJogadorMiniMax)
        
        if result == 0:
            contEmpate += 1
        elif result == 1:
            cont1 += 1
        else:
            cont2 += 1
    
    print('\nVIT PLAYER 1: ' + str(cont1))
    print('VIT PLAYER 2: ' + str(cont2))
    print('EMPATE: ' + str(contEmpate))
    
def partidaTeste():
    
    jogador1 = Jogador()
    jogador2 = Jogador()
    
    contEmpate = 0
    cont1 = 0
    cont2 = 0
    
    for i in range(0,1):
        
        ambiente = AmbienteTicTacToe(3)
        
        partida = Partida(ambiente)
        
        result = partida.partidaJogoDaVelha(jogador1.executaJogadorAleatorio, jogador2.executaJogadorMiniMax)
        
        if result == 0:
            contEmpate += 1
        elif result == 1:
            cont1 += 1
        else:
            cont2 += 1
    
    print('\nVIT PLAYER 1: ' + str(cont1))
    print('VIT PLAYER 2: ' + str(cont2))
    print('EMPATE: ' + str(contEmpate))
    
def partidaJogadores(funcaoJogador1, funcaoJogador2, numPartidas, log):
    
    contEmpate = 0
    cont1 = 0
    cont2 = 0
    
    for i in range(0,numPartidas):
        
        ambiente = AmbienteTicTacToe(3)
        
        partida = Partida(ambiente)
        
        result = partida.partidaJogoDaVelha(funcaoJogador1, funcaoJogador2, log)
        
        if result == 0:
            contEmpate += 1
        elif result == 1:
            cont1 += 1
        else:
            cont2 += 1
    
    print('\nVIT PLAYER 1: ' + str(cont1))
    print('VIT PLAYER 2: ' + str(cont2))
    print('EMPATE: ' + str(contEmpate))
    
def testePortasLogicas():
    
    redeTeste = RedeNeuralMLP(2,20,1)
    
    entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
    saidas = np.array([[0], [1], [1], [0]]) #Saída operador XOR
    
    redeTeste.executaTreinamentoModelo2Camadas(entradas, saidas)
    
    teste = np.array([[0,0], [0,1], [1,0], [1,1]])

    print(np.round(redeTeste.predicao(teste), decimals=1))

def treinamentoSupervisionadoTradicional():
    
    treinamento = TreinamentoSupervisionado(AmbienteTicTacToe(3))
    
    redeTreinada = treinamento.executaTreinamentoTradicional(numPartidas=1000, numIteracoes=20000, taxaAprendizagem=0.001, momentum=1)
    
    jogador1 = Jogador()
    jogador2 = Jogador(redeTreinada)
    
    print("\nTorneio entre Jogador Aleatório e Jogador Minimax com Rede Treinada")
    partidaJogadores(jogador1.executaJogadorAleatorio, jogador2.executaJogadorMiniMaxNN, 100, False)
    
    print("\nTorneio entre Jogador MiniMax com Função Estática e Jogador Minimax com Rede Treinada")
    partidaJogadores(jogador1.executaJogadorMiniMax, jogador2.executaJogadorMiniMaxNN, 10, False)

def main():
    
    #campeonatoTeste()
    #partidaTeste()
    #testePortasLogicas()
    
    inicio = timeit.default_timer()
    treinamentoSupervisionadoTradicional()
    total = timeit.default_timer() - inicio
    print("\nTempo Total de Execução: " + str(total))
    
    '''inicio = timeit.default_timer()
    partidaJogadores(Jogador().executaJogadorAleatorio, Jogador().executaJogadorMiniMax, 100, log=False)
    total = timeit.default_timer() - inicio
    print("\nTempo Total de Execução: " + str(total))'''
    
main()