
def time_scaling(Contaminated_data, Clean_data, standard='x'):

    """
    Time domain에서 StandardScaler transform하는 함수
    parameter: Contaminated_data, Clean_data, standard(selection: x, y, xy)
    return: scaled_Contaminated_data, scaled_Clean_data, scaler_x, scaler_y
    """

    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X = []
    y = []

    scaler_x.fit(Contaminated_data[0].reshape(-1, 1))
    scaler_y.fit(Clean_data[0].reshape(-1, 1))

    if standard == 'x':
        for xx, yy in zip(Contaminated_data, Clean_data):
            scaled_x = scaler_x.transform(xx.reshape(-1, 1))
            scaled_y = scaler_x.transform(yy.reshape(-1, 1))
            X.append(scaled_x.squeeze())
            y.append(scaled_y.squeeze())

    elif standard == 'y':
        for xx, yy in zip(Contaminated_data, Clean_data):
            scaled_x = scaler_y.transform(xx.reshape(-1, 1))
            scaled_y = scaler_y.transform(yy.reshape(-1, 1))
            X.append(scaled_x.squeeze())
            y.append(scaled_y.squeeze())
    elif standard == 'xy':
        for xx, yy in zip(Contaminated_data, Clean_data):
            scaled_x = scaler_x.transform(xx.reshape(-1, 1))
            scaled_y = scaler_y.transform(yy.reshape(-1, 1))
            X.append(scaled_x.squeeze())
            y.append(scaled_y.squeeze())

    X = np.array(X)
    y = np.array(y)

    print("X:", X.shape)
    print("y:", y.shape) 

    return X, y, scaler_x, scaler_y   

def time_inv_scaling(Contaminated, SACed, Clean, scaler_x, scaler_y=None):

    """
    Time domain에서 StandardScaler inverse transform하는 함수
    parameter: Contaminated, SACed, Clean, scaler_x, scaler_y
    return: Contaminated_inverse_scaled, SACed_inverse_scaled, Clean_inverse_scaled
    """

    import numpy as np

    Contaminated_inverse_scaled = np.array([scaler_x.inverse_transform(Contaminated[0].reshape(-1, 1)).squeeze()])
    SACed_inverse_scaled = np.array([scaler_x.inverse_transform(Contaminated[0].reshape(-1, 1)).squeeze()])
    Clean_inverse_scaled = np.array([scaler_x.inverse_transform(Contaminated[0].reshape(-1, 1)).squeeze()])

    if scaler_y == None:
        for x, y_pred, y in zip(Contaminated, SACed, Clean):
            x_inversed = scaler_x.inverse_transform(x.reshape(-1, 1)).squeeze()
            y_pred_inversed = scaler_x.inverse_transform(y_pred.reshape(-1, 1)).squeeze()
            y_inversed = scaler_x.inverse_transform(y.reshape(-1, 1)).squeeze()
            
            Contaminated_inverse_scaled = np.vstack((Contaminated_inverse_scaled, x_inversed))
            SACed_inverse_scaled = np.vstack((SACed_inverse_scaled, y_pred_inversed))
            Clean_inverse_scaled = np.vstack((Clean_inverse_scaled, y_inversed))

    else:
        for x, y_pred, y in zip(Contaminated, SACed, Clean):
            x_inversed = scaler_x.inverse_transform(x.reshape(-1, 1)).squeeze()
            y_pred_inversed = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).squeeze()
            y_inversed = scaler_y.inverse_transform(y.reshape(-1, 1)).squeeze()
            
            Contaminated_inverse_scaled = np.vstack((Contaminated_inverse_scaled, x_inversed))
            SACed_inverse_scaled = np.vstack((SACed_inverse_scaled, y_pred_inversed))
            Clean_inverse_scaled = np.vstack((Clean_inverse_scaled, y_inversed))

    Contaminated_inverse_scaled = np.delete(Contaminated_inverse_scaled, 0, axis=0)
    SACed_inverse_scaled = np.delete(SACed_inverse_scaled, 0, axis=0)
    Clean_inverse_scaled = np.delete(Clean_inverse_scaled, 0, axis=0)

    return Contaminated_inverse_scaled, SACed_inverse_scaled, Clean_inverse_scaled

def time_inv_scaling_simpleversion(Contaminated, SACed, Clean, scaler_x, scaler_y=None):
    """
    Time domain에서 StandardScaler inverse transform하는 함수
    parameter: Contaminated, SACed, Clean, scaler_x, scaler_y
    return: Contaminated_inverse_scaled, SACed_inverse_scaled, Clean_inverse_scaled
    """
    import numpy as np

    def inverse_transform(data, scaler):
        return np.array([scaler.inverse_transform(d.reshape(-1, 1)).squeeze() for d in data])

    Contaminated_inverse_scaled = inverse_transform(Contaminated, scaler_x)
    SACed_inverse_scaled = inverse_transform(SACed, scaler_x if scaler_y is None else scaler_y)
    Clean_inverse_scaled = inverse_transform(Clean, scaler_x if scaler_y is None else scaler_y)

    return Contaminated_inverse_scaled, SACed_inverse_scaled, Clean_inverse_scaled