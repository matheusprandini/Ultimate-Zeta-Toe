from RedeNeuralMLP import RedeNeuralMLP
from Jogador import Jogador
from Utilidade import Utilidade
from AmbienteTicTacToe import AmbienteTicTacToe
from Partida import Partida
import itertools
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import timeit

class TreinamentoSupervisionado:
    
    def __init__(self, ambiente):
        self.ambiente = ambiente
        
    def executaTreinamentoTradicional(self, numPartidas=1000, numIteracoes=100000, taxaAprendizagem=0.1, momentum=1):
        
        print('--- INÍCIO TREINAMENTO ---\n')
        print("Etapa 1: Criação Base de Dados -> ", numPartidas, " jogos\n")
        
        ##Criação da base de dados (entradas (estados finais) e saídas (true labels))
        baseDadosEntradasEstadosFinais = np.zeros((1,9))
        baseDadosSaidasEstadosFinais = np.zeros((1,1))
        
        iteracao = 0
        
        ##Início Etapa 1
        while iteracao < numPartidas:
            
            ##Inicializa Partida
            tabuleiro = self.ambiente.getTabuleiro().getEstadoAtual()
            
            jogadorTurno = 1
            
            #Enquanto a partida estiver sendo realizada
            while True:        
                
                movimentosLegais = list(self.ambiente.movimentosDisponiveisLegais(tabuleiro))
            
                if len(movimentosLegais) == 0:
                    break
                
                if jogadorTurno > 0:            
                    movimento = Jogador().executaJogadorAleatorio(self.ambiente, tabuleiro, 1)
                else:
                    movimento = Jogador().executaJogadorAleatorio(self.ambiente, tabuleiro, -1)
                    
                #Movimento ilegal ocasiona em vitória do oponente
                if movimento not in movimentosLegais:            
                    print("Movimento Ilegal", movimento)
                    break
                
                #Executa movimento
                tabuleiro = self.ambiente.executaMovimento(tabuleiro, movimento, jogadorTurno)        
                
                #Exibe o movimento realizado pelo jogador corrente
                #self.ambiente.exibeMovimento(tabuleiro, jogadorTurno)
                        
                #Verifica se há um vencedor
                vencedor = self.ambiente.verificaExistenciaVencedor(tabuleiro)        
                
                #Caso houver vencedor, termina a partida e atualia a melhor rede
                if vencedor != 0:
                    break
                
                #Troca turno
                jogadorTurno = -jogadorTurno
            
            ##Base de Dados de Estados Finais
            
            entrada = np.array([tabuleiro])
            saida = np.array([Utilidade(self.ambiente,'Estatica').funcaoAvaliacao(tabuleiro)]).reshape(1,1)
            
            if (saida == 1 or saida == 0 or saida == -1) and (entrada != baseDadosEntradasEstadosFinais).all(1).any():
                baseDadosEntradasEstadosFinais = np.append(baseDadosEntradasEstadosFinais, entrada, axis=0)
                baseDadosSaidasEstadosFinais = np.append(baseDadosSaidasEstadosFinais, saida, axis=0)
            else:
                iteracao -= 1

            iteracao += 1
        ##Fim Etapa 1
        
        ##Início Etapa 2
        
        print('Etapa 2: Treinamento da Rede com a Base de Dados\n')
        
        inicio = timeit.default_timer()
        rede = RedeNeuralMLP(9,20,1) #Criação da Rede
        
        ##Split da base em treinamento e teste (80% treinamento e 20% teste)
        entradasTreinamento, entradasTeste, saidasTreinamento, saidasTeste = train_test_split(baseDadosEntradasEstadosFinais, baseDadosSaidasEstadosFinais, test_size=0.2)

        custos, iteracoes = rede.executaTreinamentoModelo2Camadas(entradasTreinamento, saidasTreinamento, numIteracoes, taxaAprendizagem, momentum)
        
        plt.plot(iteracoes, custos)
        plt.xlabel('Número Iterações')
        plt.ylabel('Custo')
        plt.show()
        total = timeit.default_timer() - inicio
        
        print("Tempo de Treinamento: " + str(total) + " segundos.")
        ##Fim Etapa 2
        
        ##Início Etapa 3
        
        print("\nEtapa 3: Teste da Rede\n")
        
        ##Array de true labels e de saídas obtidas pela rede
        saidasTeste = saidasTeste.reshape(saidasTeste.shape[1], saidasTeste.shape[0])
        predicoesTeste = rede.predicao(entradasTeste)
        
        ##Arredondamento dos valores que foram obtidos pela rede (Exemplo: 0.99 = 1)
        predicoesTeste = np.round(predicoesTeste, decimals=1).reshape(saidasTeste.shape)
        
        ##Cálculo da precisão da rede
        precisaoRede = np.sum(np.equal(predicoesTeste, saidasTeste) / saidasTeste.size) * 100
        
        print ('Precisão da Rede Treinada: ', precisaoRede, '%\n')
        
        ##Fim Etapa 3
        
        return rede