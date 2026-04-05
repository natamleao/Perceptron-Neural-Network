class PerceptronTrainer:
    def train(self, model, X, y, max_epochs, logger):
        logger.start()
        samples = X.shape[0]
        converged = False
        
        for epoch in range(max_epochs):
            errors = 0
            
            for i in range(samples):
                output = model.forward(X[i])
                error = y[i,0] - output
                
                if error != 0:
                    model.update(X[i], error)
                    errors += 1

            logger.epoch(epoch + 1, model.weights, model.bias)

            if errors == 0:
                logger.convergence(epoch + 1)
                converged = True
                break
            
        if converged:
            logger.finished(model.weights, model.bias)
        else: 
            logger.max_epochs(max_epochs)