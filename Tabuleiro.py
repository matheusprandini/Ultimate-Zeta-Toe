class Tabuleiro:
    
    '''
        Atributos:
            
        - EstadoAtual: representa o estado atual do tabuleiro;
    '''
    
    #Função de Inicialização
    def __init__(self, tamanhoTabuleiro=3):
        
        '''
        Parâmetros:
            - Tamanho do Tabuleiro (Padrão: 3 (3x3));
        '''
        self.estadoAtual = self.novoTabuleiro(tamanhoTabuleiro)
    
    #Função que inicializa um novo tabuleiro (0 representa que nenhuma jogada foi realizada)
    def novoTabuleiro(self, n):
        """
        Retorno:
            - Tupla (n x n) do tipo int;
        """
        return [0] * (n * n)
    
    #Função GetEstadoAtual
    def getEstadoAtual(self):
        return self.estadoAtual