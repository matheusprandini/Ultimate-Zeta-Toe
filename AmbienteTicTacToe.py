from Tabuleiro import Tabuleiro
import itertools

class AmbienteTicTacToe:

    '''
        Atributos:
        
        - Nome: nome do ambiente, usado para sua identificação;
        - Tabuleiro: representa o tabuleiro corrente;
        - NumeroMarcacoesSequencia: representa o número de marcações que devem ser realizas em sequência
                                    para um jogador ganhar a partida;
                                    
        
        Representação do Tabuleiro: 
        
        - 0: espaço em branco (posição disponível);
        - 1: marcação do jogador que atua primeiro;
        - -1: marcação do jogador que atua segundamente;
    '''
    
    def __init__(self, tamanhoTabuleiro=3, numeroMarcacoesSequencia=3):
        self.nome = 'Jogo da Velha'
        self.tabuleiro = Tabuleiro(tamanhoTabuleiro)
        self.numeroMarcacoesSequencia = numeroMarcacoesSequencia
    
    #Função que retorna os movimentos legais de um estado corrente
    def movimentosDisponiveisLegais(self, estado):
        '''
        Argumentos:
            - Estado Atual;
        
        Retorno:
            - Lista de posições possíveis para se realizar um movimento válido;
        '''
        
        for x in range(0,9):
            if estado[x] == 0:
                yield ((int) (x / 3), (int) (x % 3))
                    
                    
    #Função que verifica se há n marcações de um mesmo lado em uma determinada reta, o que indica vitória
    def verificaNMarcacoesReta(self, reta):
        '''
        Argumentos:
            - Reta: representa uma linha, coluna ou diagonal;
            
        Retorno:
            - False: caso não haver n marcações de um mesmo símbolo em uma reta;
            - True: caso haver n marcações de um mesmo símbolo em uma reta;
        '''
        
        return all(x == -1 for x in reta) | all(x == 1 for x in reta)
    
    #Função que verifica se há um vencedor na partida
    def verificaExistenciaVencedor(self, estado):
        
        '''
        Argumento:
            - Estado: representa o estado atual;
            
        Retorno:
            - 0: não há vencedor;
            - 1: vecendor é o 'x';
            - (-1): vencedor é o 'o';
        '''
        
        #Verifica as linhas
        for x in range(0, 7, 3):
            if self.verificaNMarcacoesReta([estado[x], estado[x+1], estado[x+2]]):
                return estado[x]
        
        # check columns
        for x in range(3):
            if self.verificaNMarcacoesReta([estado[x], estado[x+3], estado[x+6]]):
                return estado[x]
    
        # check diagonals
        if self.verificaNMarcacoesReta([estado[0], estado[4], estado[8]]):
            return estado[0]
        if self.verificaNMarcacoesReta([estado[2], estado[4], estado[6]]):
            return estado[2]
    
        return 0
    
    #Função que executa uma determinada jogada no tabuleiro atual
    def executaMovimento(self, estado, movimento, marcacao):
        
        """
        Argumentos:
            - Estado: estado de tabuleiro atual (n x n);
            - Movimento: posição na qual será realizada o movimento (int, int);
            - Marcação: lado que está realizando o movimento (1, jogador 'x'; -1, jogador 'o').
        
        Retorno:
            - Novo Estado: novo estado de tabuleiro com o movimento executado.
        """
        
        move_x, move_y = movimento
    
        for x in range(0,9):
            if ((int) (x / 3) == move_x) and ((int) (x % 3) == move_y):
                temp = list(estado)
                temp[x] = marcacao
                return temp[:]

    
    ##Função que printa informações do jogador que realizou o movimento e o novo estado tabuleiro.
    def exibeMovimento(self, tabuleiro, marcacao):
        print("\n Movimento do Jogador ", marcacao)
        print(tabuleiro)
    
    #Função GetTabuleiro
    def getTabuleiro(self):
        return self.tabuleiro
    
    #Função GetNome
    def getNome(self):
        return self.nome