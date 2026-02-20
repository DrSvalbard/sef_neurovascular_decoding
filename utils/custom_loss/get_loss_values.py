def get_loss_weights(epoch, total_epochs=50):

    warmup_period = total_epochs//5
    
    w_mse = 1
    
    if epoch < warmup_period:
        w_pearson = 0.1 + (100 * (epoch / warmup_period))
        w_mse = 5.0
    else:
        w_pearson = 100.0
        w_mse = 5.0 - (3.0 * (1-epoch/total_epochs))
        
    return w_mse, w_pearson
