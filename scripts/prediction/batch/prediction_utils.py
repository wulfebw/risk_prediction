
def infer_input_dim(data):
    shape = data['x_train'].shape   
    if len(shape) == 3:
        return shape[2]
    else:
        return shape[1]

def infer_output_dim(data):
    return data['y_train'].shape[-1]
    