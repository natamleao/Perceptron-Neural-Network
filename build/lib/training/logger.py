class TrainingLogger:
    def __init__(self, enabled=True):
        self._enabled = enabled
        
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def start(self) -> None:
        if self.enabled:
            print('+' + 78*'-' + '+')
            print('+' + 28*'-' + ' Treinamento iniciado ' + 28*'-' + '+')
            print('+' + 78*'-' + '+')

    def epoch(self, epoch, weights, bias) -> None:
        if self.enabled:
            print(f'+ Ciclo: {epoch} - Pesos: {weights.flatten()} | Bias: {bias}')
            print('+' + 78*'-' + '+')

    def convergence(self, epoch) -> None:
        if self.enabled:
            print(f'+ Rede convergiu após {epoch} ciclos')
            
    def max_epochs(self, max_epochs):
        if self.enabled:
            print(f'+ Treinamento interrompido: limite de {max_epochs} épocas atingido')
            print('+' + 78*'-' + '+')

    def finished(self, weights, bias) -> None:
        if self.enabled:
            print('+' + 78*'-' + '+')
            print('+' + 27*'-' + ' Treinamento finalizado ' + 27*'-' + '+')
            print('+' + 78*'-' + '+')
            print(f'+ Pesos finais: {weights.flatten()} | Bias final: {bias}')
            print('+' + 78*'-' + '+')