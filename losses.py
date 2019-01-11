from keras import backend as K

def grid_loss(y_true, y_pred):
    true_boxes = y_true[:,:,:,0]
    pred_boxes = y_pred[:,:,:,0]
    box_loss = K.binary_crossentropy(true_boxes, pred_boxes)
    pos_xloss = true_boxes * K.binary_crossentropy(y_true[:,:,:,1], y_pred[:,:,:,1])
    pos_yloss = true_boxes * K.binary_crossentropy(y_true[:,:,:,2], y_pred[:,:,:,2])
    return box_loss + pos_xloss + pos_yloss

def grid_loss_with_hands(y_true, y_pred):
    ball_loss = grid_loss(y_true[:,:,:,0:3], y_pred[:,:,:,0:3])
    rhand_loss = grid_loss(y_true[:,:,:,3:6], y_pred[:,:,:,3:6])
    lhand_loss = grid_loss(y_true[:,:,:,6:9], y_pred[:,:,:,6:9])
    return ball_loss + rhand_loss + lhand_loss
