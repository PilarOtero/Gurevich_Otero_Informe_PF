import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.metrics import rmse, mae, r2

def get_activaciones(nombre):
    nombre= nombre.lower()

    if nombre == "relu" :
        return nn.ReLU()
    elif nombre == "leakyrelu":
        return nn.LeakyReLU(negative_slope= 0.01)
    elif nombre == "gelu" :
        return nn.GELU()
    elif nombre == "silu" : #en pytorch es swish
        return nn.SiLU()
    elif nombre == "swish" : 
        return nn.SiLU()
    else:
        return ValueError("ACTIVACION NO SOPORTADA")
    
class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        capas_ocultas: list[int],
        activacion: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False
    ):
        super().__init__()

        capas = []
        dim_actual = in_dim

        for h in capas_ocultas:
            capas.append(nn.Linear(dim_actual, h))

            if batchnorm:
                capas.append(nn.BatchNorm1d(h))

            capas.append(get_activaciones(activacion))

            if dropout > 0:
                capas.append(nn.Dropout(dropout))

            dim_actual = h

        capas.append(nn.Linear(dim_actual, 1))

        self.net = nn.Sequential(*capas)

    def forward(self, x):
        return self.net(x).squeeze(1)


def evaluar_MLP(modelo, loader, criterio, device):
    modelo.eval()

    total_loss = 0.0
    total = 0

    y_true_log_all = []
    y_pred_log_all = []

    with torch.no_grad():
        for X_batch, y_batch_log in loader:
            X_batch = X_batch.to(device)
            y_batch_log = y_batch_log.to(device)

            y_pred_log = modelo(X_batch)
            loss = criterio(y_pred_log, y_batch_log)

            total_loss += loss.item() * X_batch.size(0)
            total += X_batch.size(0)

            y_true_log_all.extend(y_batch_log.cpu().numpy())
            y_pred_log_all.extend(y_pred_log.cpu().numpy())

    loss_log = total_loss / total

    y_true_log_all = np.array(y_true_log_all)
    y_pred_log_all = np.array(y_pred_log_all)

    y_true = np.expm1(y_true_log_all)
    y_pred = np.expm1(y_pred_log_all)
    y_pred = np.clip(y_pred, 0, None)

    return {
        "loss_log": loss_log,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "y_pred": y_pred
    }

def entrenar_red_neuronal(
    X_train,
    y_train,
    X_val,
    y_val,
    capas_ocultas=[256, 128],
    activacion="relu",
    dropout=0.2,
    batchnorm=True,
    epochs=200,
    batch_size=256,
    alpha=1e-3,
    scheduler=None,
    c=0.8,
    s=40,
    alpha_min=1e-5,
    weight_decay=1e-4,
    usar_early_stop=True,
    paciencia=20,
    min_delta=1e-4,
    gradient_clip=1.0,
    random_state=42,
    mostrar=False
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_np = np.asarray(X_train, dtype=np.float32)
    X_val_np = np.asarray(X_val, dtype=np.float32)

    y_train_log = np.log1p(np.asarray(y_train, dtype=np.float32))
    y_val_log = np.log1p(np.asarray(y_val, dtype=np.float32))

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_log, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_log, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False
    )

    modelo = MLP(
        in_dim=X_train_np.shape[1],
        capas_ocultas=capas_ocultas,
        activacion=activacion,
        dropout=dropout,
        batchnorm=batchnorm
    ).to(device)

    criterio = nn.MSELoss() #FUNCION DE LOSS PARA regresion-> mide error cuadratico en escala log

    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=alpha,
        weight_decay=weight_decay
    )

    train_losses = []
    val_losses = []
    val_rmse = []
    val_mae = []
    val_r2 = []
    alphas = []

    mejor_val_loss = np.inf
    mejores_pesos = None
    contador = 0

    inicio = time.time()

    for epoch in range(epochs):

        if scheduler == "expo":
            alpha_epoch = max(alpha * (c ** (epoch / s)), alpha_min)
        else:
            alpha_epoch = alpha

        for grupo in optimizador.param_groups:
            grupo["lr"] = alpha_epoch

        alphas.append(alpha_epoch)

        modelo.train()

        for X_batch, y_batch_log in train_loader:
            X_batch = X_batch.to(device)
            y_batch_log = y_batch_log.to(device)

            optimizador.zero_grad()

            y_pred_log = modelo(X_batch)
            loss = criterio(y_pred_log, y_batch_log)

            loss.backward()

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    modelo.parameters(),
                    gradient_clip
                )

            optimizador.step()

        train_eval = evaluar_MLP(modelo, train_loader, criterio, device)
        val_eval = evaluar_MLP(modelo, val_loader, criterio, device)

        train_losses.append(train_eval["loss_log"])
        val_losses.append(val_eval["loss_log"])
        val_rmse.append(val_eval["rmse"])
        val_mae.append(val_eval["mae"])
        val_r2.append(val_eval["r2"])

        if mostrar:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"alpha={alpha_epoch:.6f} | "
                f"Train Loss={train_eval['loss_log']:.5f} | "
                f"Val Loss={val_eval['loss_log']:.5f} | "
                f"RMSE={val_eval['rmse']:.2f} | "
                f"MAE={val_eval['mae']:.2f} | "
                f"R2={val_eval['r2']:.4f}"
            )

        if usar_early_stop:
            if val_eval["loss_log"] < mejor_val_loss - min_delta:
                mejor_val_loss = val_eval["loss_log"]
                mejores_pesos = copy.deepcopy(modelo.state_dict())
                contador = 0
            else:
                contador += 1

            if contador >= paciencia:
                if mostrar:
                    print(f"Early stopping en epoch {epoch + 1}")
                modelo.load_state_dict(mejores_pesos)
                break

    tiempo = time.time() - inicio

    final_eval = evaluar_MLP(modelo, val_loader, criterio, device)

    historial = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "alpha": alphas,
        "train_loss_log": train_losses,
        "val_loss_log": val_losses,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2
    })

    resultados = {
        "RMSE": final_eval["rmse"],
        "MAE": final_eval["mae"],
        "R2": final_eval["r2"],
        "tiempo": tiempo
    }

    return modelo, final_eval["y_pred"], historial, resultados


def buscar_mejor_red_neuronal(
    X_train,
    y_train,
    X_val,
    y_val,
    configuraciones: list[dict],
    epochs=200,
    batch_size=256,
    random_state=42
):
    resultados = []
    modelos = {}
    historiales = {}
    predicciones = {}

    for i, config in enumerate(configuraciones):
        nombre = config.get("nombre", f"NN_{i + 1}")

        print(f"\nEntrenando {nombre}")
        print(config)

        modelo, y_pred, historial, metricas = entrenar_red_neuronal(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            capas_ocultas=config.get("capas_ocultas", [256, 128]),
            activacion=config.get("activacion", "relu"),
            dropout=config.get("dropout", 0.2),
            batchnorm=config.get("batchnorm", True),
            alpha=config.get("alpha", 1e-3),
            scheduler=config.get("scheduler", "expo"),
            c=config.get("c", 0.8),
            s=config.get("s", 40),
            alpha_min=config.get("alpha_min", 1e-5),
            weight_decay=config.get("weight_decay", 1e-4),
            gradient_clip=config.get("gradient_clip", 1.0),
            paciencia=config.get("paciencia", 20),
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            mostrar=False
        )

        resultados.append({
            "Modelo": nombre,
            "capas_ocultas": config.get("capas_ocultas", [256, 128]),
            "activacion": config.get("activacion", "relu"),
            "dropout": config.get("dropout", 0.2),
            "batchnorm": config.get("batchnorm", True),
            "alpha": config.get("alpha", 1e-3),
            "scheduler": config.get("scheduler", "expo"),
            "weight_decay": config.get("weight_decay", 1e-4),
            "RMSE": round(metricas["RMSE"], 2),
            "MAE": round(metricas["MAE"], 2),
            "R2": round(metricas["R2"], 4),
            "tiempo": round(metricas["tiempo"], 2)
        })

        modelos[nombre] = modelo
        historiales[nombre] = historial
        predicciones[nombre] = y_pred

    resultados_df = (
        pd.DataFrame(resultados)
        .sort_values("RMSE", ascending=True)
        .reset_index(drop=True)
    )

    return resultados_df, modelos, historiales, predicciones