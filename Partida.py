from RedeNeuralMLP import RedeNeuralMLP

class Partida:
    
    '''
    Atributos:
        
        - Ambiente: representa em qual ambiente (tipo de jogo) a partida irá ocorrer;
    '''
    
    def __init__(self, ambiente):
        self.ambiente = ambiente
    
    #Função que controla uma partida de jogo da velha 
    def partidaJogoDaVelha(self, funcaoJogador1, funcaoJogador2, log=False):
        
        '''
        Parâmetros:
            
            - Função Jogador 1: método de execução de movimentos do jogador 1 (representa 1);
            - Função Jogador 2: método de execução de movimentos do jogador 2 (representa -1);
            
        Retorno:
            
            - 0: Empate;
            - 1: Vencedor é Jogador 1;
            - -1: Vencedor é Jogador 2;
        '''
        
        #Recupera o estado atual do tabuleiro
        tabuleiro = self.ambiente.getTabuleiro().getEstadoAtual()
        
        jogadorTurno = 1 #Representa o '1'
        
        #Enquanto a partida estiver sendo realizada
        while True:        
            
            #Verifica se há movimentos disponíveis a serem realizados no atual estado de tabuleiro
            movimentosLegais = list(self.ambiente.movimentosDisponiveisLegais(tabuleiro))        
           
            #Empate
            if len(movimentosLegais) == 0:            
                if log:    
                    print("\nEmpate")            
                return 0
            
            #Chama métodos dos jogadores
            if jogadorTurno > 0:            
                movimento = funcaoJogador1(self.ambiente, tabuleiro, 1)  
            else:
                movimento = funcaoJogador2(self.ambiente, tabuleiro, -1)
         
            #Movimento ilegal ocasiona em vitória do oponente
            if movimento not in movimentosLegais:            
                #print("\nMovimento Ilegal", movimento)            
                return -jogadorTurno
            
            #Executa movimento
            tabuleiro = self.ambiente.executaMovimento(tabuleiro, movimento, jogadorTurno)        
            
            if log:
                #Exibe o movimento realizado pelo jogador corrente
                self.ambiente.exibeMovimento(tabuleiro, jogadorTurno)
            
            #Verifica se há um vencedor
            vencedor = self.ambiente.verificaExistenciaVencedor(tabuleiro)        
            
            #Caso houver vencedor, termina a partida
            if vencedor != 0:      
                if log:
                    print("\nVencedor: %s" % jogadorTurno)            
                return vencedor        
            
            #Troca turno
            jogadorTurno = -jogadorTurno