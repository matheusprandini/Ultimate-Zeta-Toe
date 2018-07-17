from numpy import inf
from Utilidade import Utilidade

class BuscaMinimax:
    
    '''
    Atributos:
        
        - Ambiente: representa o tipo de jogo em que a busca minimax irá atuar;
    '''
    
    def __init__(self, ambiente, tipoUtilidade='Estatica', rede=None):
        self.ambiente = ambiente
        self.funcaoUtilidade = Utilidade(ambiente, tipoUtilidade, rede).funcaoAvaliacao
    
    #Função de Busca Competitiva Minimax
    def algoritmoMinimax(self, ambiente, estado, marcacao, profundidadeMaxima):
        if marcacao > 0:
            return self.algoritmoMax(ambiente, estado, marcacao, profundidadeMaxima)
        else:
            return self.algoritmoMin(ambiente, estado, marcacao, profundidadeMaxima)
    
    #Função Max
    def algoritmoMax(self, ambiente, estado, marcacao, profundidadeMaxima):
         
       #Carrega todos os movimentos legais disponíveis
        movimentos = list(ambiente.movimentosDisponiveisLegais(estado))

        if not movimentos: #Nenhum movimento disponível (CASO DE EMPATE -> RETORNO 0)
            return self.funcaoUtilidade(estado), None
        
        melhorResultado = -inf #Inicializa melhor resultado como menos infinito
        melhorMovimento = None
        
        #Realização de todos os movimentos disponíveis e armazena todos os novos estados gerados
        for movimento in movimentos:
            novoEstado = ambiente.executaMovimento(estado, movimento, marcacao)
            
            #Verifica se o novo estado contém um vencedor 
            vencedor = ambiente.verificaExistenciaVencedor(novoEstado)
            
            #Caso sim, retorna como melhor valor e o movimento que deve ser feito para ganhar ou evitar derrota
            if vencedor != 0:
                return self.funcaoUtilidade(novoEstado), movimento
            
            #Caso não haja vencedor
            else:
                
                ##Nó terminal -> Chama Função de utilidade do estado (Função de Avaliação)
                if profundidadeMaxima <= 1:
                    resultado = self.funcaoUtilidade(novoEstado)
                else: #Chama função Min para os filhos
                    resultado, _ = self.algoritmoMin(ambiente, novoEstado, -marcacao, profundidadeMaxima - 1)
                
                #Verifica se o resultado/movimento é o melhor possível
                if resultado > melhorResultado:
                    melhorResultado = resultado
                    melhorMovimento = movimento
                        
        return melhorResultado, melhorMovimento
    
    #Função Min
    def algoritmoMin(self, ambiente, estado, marcacao, profundidadeMaxima):
         
        #Carrega todos os movimentos legais disponíveis
        movimentos = list(ambiente.movimentosDisponiveisLegais(estado))
        
        if not movimentos: #Nenhum movimento disponível (CASO DE EMPATE -> RETORNO 0)
            return self.funcaoUtilidade(estado), None
        
        melhorResultado = inf #Inicializa melhor resultado como mais infinito
        melhorMovimento = None
        
        #Realização de todos os movimentos disponíveis e armazena todos os novos estados gerados
        for movimento in movimentos:
            novoEstado = ambiente.executaMovimento(estado, movimento, marcacao)
            
            #Verifica se o novo estado contém um vencedor 
            vencedor = ambiente.verificaExistenciaVencedor(novoEstado)
            
            #Caso sim, retorna como melhor valor e o movimento que deve ser feito para ganhar ou evitar derrota
            if vencedor != 0:
                return self.funcaoUtilidade(novoEstado), movimento
            
            #Caso não haja vencedor
            else:
                
                ##Nó terminal -> Chama Função de utilidade do estado (Função de Avaliação)
                if profundidadeMaxima <= 1:
                    resultado = self.funcaoUtilidade(novoEstado)
                else: #Chama função Max para os filhos
                    resultado, _ = self.algoritmoMax(ambiente, novoEstado, -marcacao, profundidadeMaxima - 1)
                
                #Verifica se o resultado/movimento é o melhor possível
                if resultado < melhorResultado:
                    melhorResultado = resultado
                    melhorMovimento = movimento
                        
        return melhorResultado, melhorMovimento