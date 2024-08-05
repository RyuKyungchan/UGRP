
def time_scaling(Contaminated_data, Clean_data):

    """
    Time domain에서 StandardScaler transform하는 함수
    parameter: Contaminated_data, Clean_data
    return: scaled_Contaminated_data, scaled_Clean_data
    """

    

def time_inv_scaling(Contaminated, SACed, Clean, scaler_x, scaler_y):

    """
    Time domain에서 StandardScaler inverse transform하는 함수
    parameter: Contaminated, SACed, Clean, scaler_x, scaler_y
    return: Contaminated_inverse_scaled, SACed_inverse_scaled, Clean_inverse_scaled
    """

    import numpy as np

    Contaminated_inverse_scaled = np.array([scaler_x.inverse_transform(Contaminated[0].reshape(-1, 1)).squeeze()])
    SACed_inverse_scaled = np.array([scaler_x.inverse_transform(Contaminated[0].reshape(-1, 1)).squeeze()])
    Clean_inverse_scaled = np.array([scaler_x.inverse_transform(Contaminated[0].reshape(-1, 1)).squeeze()])

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