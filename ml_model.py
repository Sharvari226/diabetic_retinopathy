# ml_model_v2_explainable.py
"""
Enhanced Multi-Task DR Screening Model
- EfficientNetV2B0 + CBAM Attention
- Mixed precision training
- Image Quality–Aware multitask loss
- Focal loss for DR imbalance
- Grad-CAM support for explainability
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, roc_auc_score

# ---------- CONFIG ----------
DATA_DIR = r'C:\Users\SHARVARI JADHAV\Downloads\Diabetic Retinopathy\processed'
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IQA_WEIGHT = 0.3
DR_WEIGHT = 1.0
SEED = 42
IMG_SHAPE = (224, 224, 1)
MIXED_PRECISION = True
# ----------------------------

# ---------- SETUP ----------
tf.random.set_seed(SEED)
np.random.seed(SEED)
if MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

# ---------- CBAM BLOCK ----------
def cbam_block(input_tensor, reduction_ratio=8):
    ch = input_tensor.shape[-1]
    shared_dense1 = layers.Dense(ch // reduction_ratio, activation='relu', kernel_initializer='he_normal')
    shared_dense2 = layers.Dense(ch, kernel_initializer='he_normal')

    # ---- Channel attention ----
    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    max_pool = layers.GlobalMaxPooling2D()(input_tensor)

    avg_fc = shared_dense2(shared_dense1(layers.Reshape((1,1,ch))(avg_pool)))
    max_fc = shared_dense2(shared_dense1(layers.Reshape((1,1,ch))(max_pool)))
    ch_att = layers.Add()([avg_fc, max_fc])
    ch_att = layers.Activation('sigmoid')(ch_att)
    ch_refined = layers.Multiply()([input_tensor, ch_att])

    # ---- Spatial attention (wrapped safely in Lambda layers) ----
    avg_sp = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(ch_refined)
    max_sp = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(ch_refined)
    concat = layers.Concatenate(axis=3)([avg_sp, max_sp])
    sp_att = layers.Conv2D(1, 7, padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat)

    return layers.Multiply()([ch_refined, sp_att])

# ---------- FOCAL LOSS ----------
def binary_focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        ce = -y_true*K.log(y_pred) - (1-y_true)*K.log(1-y_pred)
        w = alpha*y_true*K.pow(1-y_pred, gamma) + (1-alpha)*(1-y_true)*K.pow(y_pred, gamma)
        return K.mean(w*ce)
    return loss

# ---------- METRICS ----------
def sensitivity(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred)
    pos = K.sum(y_true)
    return tp / (pos + K.epsilon())

def specificity(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_pred = K.round(y_pred)
    tn = K.sum((1 - y_true) * (1 - y_pred))
    neg = K.sum(1 - y_true)
    return tn / (neg + K.epsilon())


# ---------- MODEL ----------
def build_multitask_model():
    inp = layers.Input(shape=(224,224,3))  # ✅ change from (224,224,1)
    base = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inp)
    base.trainable = True

    feat = base.output
    att = cbam_block(feat)
    merged = layers.Concatenate()([
        layers.GlobalAveragePooling2D()(feat),
        layers.GlobalAveragePooling2D()(att)
    ])
    x = layers.Dense(512, activation='relu')(merged)
    x = layers.Dropout(0.4)(x)

    dr = layers.Dense(128, activation='relu')(x)
    dr = layers.Dropout(0.3)(dr)
    dr_out = layers.Dense(1, activation='sigmoid', name='dr_output')(dr)

    iq = layers.Dense(128, activation='relu')(x)
    iq = layers.Dropout(0.3)(iq)
    iq_out = layers.Dense(1, activation='sigmoid', name='iqa_output')(iq)

    return models.Model(inp, [dr_out, iq_out])


# ---------- LOAD DATA ----------
print("Loading data from:", DATA_DIR)
images = np.load(os.path.join(DATA_DIR, "images.npy"), allow_pickle=True)
names  = np.load(os.path.join(DATA_DIR, "image_names.npy"), allow_pickle=True)
labels = np.load(os.path.join(DATA_DIR, "labels.npy"), allow_pickle=True).astype(int)
assert len(images)==len(names)==len(labels), "Mismatch in image/name/label count!"

if images.ndim == 3: images = np.expand_dims(images, -1)
images = images.astype("float32") / 255.0

iqa_labels = np.ones(len(labels))
for i,n in enumerate(names):
    if any(w in str(n).lower() for w in ['blur','bad','poor','lowq']):
        iqa_labels[i] = 0

from sklearn.model_selection import train_test_split
Xtr,Xte,ytr_cls,yte_cls,ytr_iqa,yte_iqa = train_test_split(
    images, labels, iqa_labels, test_size=0.2, stratify=labels, random_state=SEED
)

# ---------- AUGMENTATION ----------
aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05),
])

def gen(X,yc,yi,batch=BATCH_SIZE,aug_on=True):
    n=len(X); idx=np.arange(n)
    while True:
        np.random.shuffle(idx)
        for i in range(0,n,batch):
            b=idx[i:i+batch]; imgs=X[b]
            if aug_on: imgs=aug(imgs,training=True)
            yield imgs,{"dr_output":yc[b].reshape(-1,1),"iqa_output":yi[b].reshape(-1,1)}

# ---------- TRAIN ----------
model = build_multitask_model()
model.load_weights("best_dr_model_v2.h5")   # reuse trained weights
model.save("dr_multitask_v2_final", save_format="tf")

model.compile(
    optimizer=Adam(LEARNING_RATE),
    loss={"dr_output":binary_focal_loss(), "iqa_output":"binary_crossentropy"},
    loss_weights={"dr_output":DR_WEIGHT,"iqa_output":IQA_WEIGHT},
    metrics={"dr_output":["AUC",sensitivity,specificity],"iqa_output":["accuracy"]}
)
model.summary()

callbacks=[
    ModelCheckpoint("best_dr_model_v2.h5", monitor="val_dr_output_auc", mode="max", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_dr_output_loss", factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor="val_dr_output_auc", patience=7, mode="max", restore_best_weights=True, verbose=1)
]

train_g=gen(Xtr,ytr_cls,ytr_iqa)
val_g  =gen(Xte,yte_cls,yte_iqa,aug_on=False)
steps=len(Xtr)//BATCH_SIZE
val_steps=len(Xte)//BATCH_SIZE

# Example class weighting if imbalance exists
class_weight_dr = {0: 1.0, 1: 2.0}

history=model.fit(
    train_g, validation_data=val_g,
    steps_per_epoch=steps, validation_steps=val_steps,
    epochs=EPOCHS, verbose=2, callbacks=callbacks,
#    class_weight={"dr_output": class_weight_dr}
)

# ---------- EVALUATION ----------
pred_dr,pred_iqa=model.predict(Xte,batch_size=BATCH_SIZE)
pred_dr=pred_dr.ravel(); pred_iqa=pred_iqa.ravel()
mask=pred_iqa>=0.5

def metrics(y,p,t=0.5):
    pred=(p>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
    sens=tp/(tp+fn+1e-8); spec=tn/(tn+fp+1e-8)
    auc=roc_auc_score(y,p)
    return {"sensitivity":sens,"specificity":spec,"auc":auc}

print("\n=== Full Test Set ===")
print(metrics(yte_cls,pred_dr))
if mask.sum()>0:
    print(f"\n=== IQA Filtered (good only: {mask.sum()} images) ===")
    print(metrics(yte_cls[mask],pred_dr[mask]))

model.save("dr_multitask_v2_final.h5")
print("\n✅ Model saved as dr_multitask_v2_final.h5")

# ---------- GRAD-CAM SUPPORT ----------
def grad_cam(model, img, layer_name='top_conv'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output[0]])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img, 0))
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam /= cam.max() + 1e-8
    return cam

# save in TensorFlow SavedModel format – no Lambda deserialization issues
model.save("dr_multitask_v2_final", save_format="tf")
print("✅ SavedModel exported to dr_multitask_v2_final/")


