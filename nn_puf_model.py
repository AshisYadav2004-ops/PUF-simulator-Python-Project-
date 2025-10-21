
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from puf_simulator import ArbiterPUF, XORArbiterPUF, generate_dataset 


def build_nn(input_dim: int, hidden_units: list[int] = [64, 32]) -> tf.keras.Model:
    
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(0.2))  

    model.add(layers.Dense(1, activation="sigmoid"))  
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model



def train_puf_nn(model, challenges, responses, epochs=20, batch_size=32, test_split=0.2):

    n = len(challenges)
    split_idx = int(n * (1 - test_split))
    X_train, X_test = challenges[:split_idx], challenges[split_idx:]
    y_train, y_test = responses[:split_idx], responses[split_idx:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final test accuracy: {test_acc:.4f}")

    return model, history



def demo_nn_attack():
    
    k = 16           
    n = 10000        
    xor_degree = 1   
    seed = 12345

    
    if xor_degree == 1:
        puf = ArbiterPUF(k, seed=seed)
    else:
        puf = XORArbiterPUF(xor_degree, k, seed=seed)

    
    challenges, responses = generate_dataset(puf, n=n, k=k, seed=seed + 1, noise_std=0.0)
    print(f"Generated {n} CRPs for {'XOR' if xor_degree>1 else 'Arbiter'}-PUF")

    
    model = build_nn(input_dim=k, hidden_units=[64, 32])
    model, history = train_puf_nn(model, challenges, responses, epochs=15)

    
    new_ch, new_resp = generate_dataset(puf, n=1000, k=k, seed=seed + 2)
    preds = (model.predict(new_ch) > 0.5).astype(int).flatten()
    acc = np.mean(preds == new_resp)
    print(f"Generalization accuracy on unseen CRPs: {acc:.4f}")



if __name__ == "__main__":
    demo_nn_attack()
