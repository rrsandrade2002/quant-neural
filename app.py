# =================================================================================================================================================================#
# QUANT NEURAL SAAS - PLATAFORMA INTEGRADA DE AUDITORIA DE REDES NEURAIS (MONÓLITO ÚNICO)                                                                         #
# Autor: Rafael Rocha Siqueira de Andrade                                                                                                                               #
# =================================================================================================================================================================#

import os
import gc
from datetime import datetime
from typing import List, Dict

import cv2
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregamento do interpretador compatível com Python 3.13
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import uvicorn

# --------------------------------------------------------------------------------------------------
# 1. BANCO DE DADOS RELACIONAL (SQLite + SQLAlchemy)
# --------------------------------------------------------------------------------------------------
DATABASE_URL = "sqlite:///./quant_saas.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AuditoriaHistorico(Base):
    __tablename__ = "auditorias"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), index=True, nullable=False)
    acuracia = Column(Float, nullable=False)
    amostras_teste = Column(Integer, nullable=False)
    executado_em = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------------------------------------------------------------------------------------------
# 2. ESQUEMAS DE DADOS (Pydantic)
# --------------------------------------------------------------------------------------------------
class AuditoriaRequisicao(BaseModel):
    ticker: str

class AuditoriaResposta(BaseModel):
    ticker: str
    acuracia: float
    total_amostras: int
    distribuicao_real: Dict[str, int]
    distribuicao_prevista: Dict[str, int]
    matriz_confusao: List[List[int]]

class HistoricoResposta(BaseModel):
    id: int
    ticker: str
    acuracia: float
    amostras_teste: int
    executado_em: datetime

    class Config:
        from_attributes = True

# --------------------------------------------------------------------------------------------------
# 3. MOTOR DE PROCESSAMENTO QUANTITATIVO, CV2 E TFLITE
# --------------------------------------------------------------------------------------------------
def calcular_indicadores(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    close = d['Close']
    d['ma20'] = close.rolling(20).mean()
    d['vol_20'] = close.rolling(20).std()
    d['ret_5'] = close.pct_change(5) * 100
    
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d['rsi'] = 100.0 - (100.0 / (1.0 + rs.fillna(0)))
    return d.dropna()

def executar_auditoria_cv2(ticker: str) -> dict:
    caminho_tflite = f"analista_triplo_{ticker}_otimizado.tflite"
    caminho_scaler = f"scaler_triplo_{ticker}.gz"

    if not os.path.exists(caminho_tflite):
        caminho_tflite = f"analista_triplo_{ticker}.tflite"

    if not os.path.exists(caminho_tflite) or not os.path.exists(caminho_scaler):
        raise FileNotFoundError(f"Arquivos do modelo ({caminho_tflite} ou {caminho_scaler}) ausentes no servidor.")

    SEQ_LENGTH, FUTURE_DAYS, THRESHOLD = 60, 5, 0.04
    FEATURES = ['Close', 'ma20', 'vol_20', 'ret_5', 'rsi', 'Volume']

    df = yf.download(ticker, period='max', interval='1d', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < (SEQ_LENGTH + FUTURE_DAYS + 50):
        raise ValueError(f"Histórico de preços insuficiente para o ativo {ticker}.")

    dados = calcular_indicadores(df)
    scaler = joblib.load(caminho_scaler)
    
    colunas_usadas = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else FEATURES

    dados_norm = pd.DataFrame(scaler.transform(dados[colunas_usadas]), columns=colunas_usadas, index=dados.index)

    X_seq, X_img, y = [], [], []
    valores_close = dados['Close'].values
    valores_norm = dados_norm.values
    total_linhas = len(dados_norm)

    for i in range(SEQ_LENGTH, total_linhas - FUTURE_DAYS):
        X_seq.append(valores_norm[i-SEQ_LENGTH:i])

        retorno = (valores_close[i + FUTURE_DAYS] - valores_close[i]) / valores_close[i]
        label = 0 if retorno > THRESHOLD else (1 if retorno < -THRESHOLD else 2)
        y.append(label)

        janela = valores_close[i-SEQ_LENGTH:i]
        min_p, max_p = janela.min(), janela.max()

        if max_p - min_p == 0:
            y_pts = np.full(SEQ_LENGTH, 64, dtype=np.int32)
        else:
            norm = (janela - min_p) / (max_p - min_p)
            y_pts = np.round(122 - (norm * 117)).astype(np.int32)

        x_pts = np.linspace(0, 127, SEQ_LENGTH).astype(np.int32)
        pontos = np.column_stack((x_pts, y_pts)).reshape((-1, 1, 2))

        img = np.full((128, 128, 3), 255, dtype=np.uint8)
        cv2.polylines(img, [pontos], isClosed=False, color=(0, 0, 0), thickness=2)
        X_img.append(img.astype(np.float32) / 255.0)

    X_seq = np.array(X_seq, dtype=np.float32)
    X_img = np.array(X_img, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    split_idx = int(len(y) * 0.8)
    X_img_val, X_seq_val, y_val = X_img[split_idx:], X_seq[split_idx:], y[split_idx:]

    interpreter = Interpreter(model_path=caminho_tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    idx_img, idx_seq = None, None
    for detail in input_details:
        shape = list(detail['shape'])
        if len(shape) == 4 and shape[1:] == [128, 128, 3]:
            idx_img = detail['index']
        elif len(shape) == 3 and shape[1:] == [SEQ_LENGTH, len(colunas_usadas)]:
            idx_seq = detail['index']

    if idx_img is None or idx_seq is None:
        idx_img = input_details[0]['index']
        idx_seq = input_details[1]['index']

    y_pred = []
    for idx in range(len(y_val)):
        interpreter.set_tensor(idx_img, np.expand_dims(X_img_val[idx], axis=0))
        interpreter.set_tensor(idx_seq, np.expand_dims(X_seq_val[idx], axis=0))
        interpreter.invoke()
        saida = interpreter.get_tensor(output_details[0]['index'])[0]
        y_pred.append(int(np.argmax(saida)))

    y_pred = np.array(y_pred, dtype=np.int32)
    classes = ['COMPRA (ALTA)', 'VENDA (BAIXA)', 'MANTER']
    
    dist_real = {classes[k]: int(v) for k, v in zip(*np.unique(y_val, return_counts=True))}
    dist_pred = {classes[k]: int(v) for k, v in zip(*np.unique(y_pred, return_counts=True))}

    acuracia = float(accuracy_score(y_val, y_pred))
    matriz = confusion_matrix(y_val, y_pred, labels=[0, 1, 2]).tolist()

    gc.collect()

    return {
        "ticker": ticker,
        "acuracia": round(acuracia, 4),
        "total_amostras": int(len(y_val)),
        "distribuicao_real": dist_real,
        "distribuicao_prevista": dist_pred,
        "matriz_confusao": matriz
    }

# --------------------------------------------------------------------------------------------------
# 4. FRONTEND EMBUTIDO (HTML5 + CSS3 + JAVASCRIPT COM ACESSIBILIDADE WCAG)
# --------------------------------------------------------------------------------------------------
HTML_EMBUTIDO = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Neural SaaS - Auditoria de Modelos</title>
    <style>
        :root {
            --bg-color: #f1f5f9;
            --text-color: #0f172a;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --card-bg: #ffffff;
            --border: #cbd5e1;
            --focus-ring: #f59e0b;
        }
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 1rem;
            line-height: 1.5;
        }
        header, footer {
            text-align: center;
            max-width: 800px;
            margin: 0 auto;
            padding: 1rem 0;
        }
        main {
            max-width: 800px;
            margin: 0 auto;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .input-group {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        input[type="text"] {
            flex: 1;
            padding: 0.75rem;
            font-size: 1rem;
            border: 1px solid var(--border);
            border-radius: 4px;
        }
        button {
            background-color: var(--primary);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:focus, input:focus {
            outline: 3px solid var(--focus-ring);
            outline-offset: 2px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .grid-metricas {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .metrica-box {
            background: #e2e8f0;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
        }
        .metrica-box .valor {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        th, td {
            border: 1px solid var(--border);
            padding: 0.75rem;
            text-align: center;
        }
        th {
            background-color: #f8fafc;
        }
        .hidden { display: none; }
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            border: 0;
        }
    </style>
</head>
<body>
    <header role="banner">
        <h1>Quant Neural SaaS</h1>
        <p>Motor de Auditoria Multimodal (OpenCV + TFLite)</p>
    </header>

    <main role="main">
        <section class="card" aria-labelledby="form-heading">
            <h2 id="form-heading">Executar Auditoria</h2>
            <form id="form-auditoria" novalidate>
                <label for="ticker-input">Código do Ativo:</label>
                <div class="input-group">
                    <input 
                        type="text" 
                        id="ticker-input" 
                        name="ticker" 
                        placeholder="Ex: AGRO3.SA, PETR4.SA" 
                        required 
                        aria-required="true"
                    >
                    <button type="submit" id="btn-submit">Processar</button>
                </div>
            </form>
            <div id="status-regiao" aria-live="assertive" class="sr-only"></div>
        </section>

        <section id="area-resultados" class="card hidden" aria-labelledby="resultado-heading">
            <h2 id="resultado-heading">Resultado da Auditoria</h2>
            <div class="grid-metricas">
                <div class="metrica-box">
                    <p>Acurácia Geral</p>
                    <p id="res-acuracia" class="valor">0.00%</p>
                </div>
                <div class="metrica-box">
                    <p>Amostras Validadas</p>
                    <p id="res-amostras" class="valor">0</p>
                </div>
            </div>

            <h3>Matriz de Confusão</h3>
            <table aria-label="Matriz de confusão">
                <thead>
                    <tr>
                        <th scope="col">Real \ Previsto</th>
                        <th scope="col">Compra (Alta)</th>
                        <th scope="col">Venda (Baixa)</th>
                        <th scope="col">Manter</th>
                    </tr>
                </thead>
                <tbody id="matriz-corpo"></tbody>
            </table>
        </section>

        <section class="card" aria-labelledby="hist-heading">
            <h2 id="hist-heading">Histórico no Banco de Dados</h2>
            <button id="btn-hist" type="button">Recarregar Histórico</button>
            <ul id="lista-historico" aria-label="Lista de auditorias salvas"></ul>
        </section>
    </main>

    <footer role="contentinfo">
        <p>&copy; 2026 Quant Neural SaaS. Interface Acessível (WCAG).</p>
    </footer>

    <script>
        document.addEventListener("DOMContentLoaded", () => {
            const form = document.getElementById("form-auditoria");
            const tickerInput = document.getElementById("ticker-input");
            const btnSubmit = document.getElementById("btn-submit");
            const statusRegiao = document.getElementById("status-regiao");
            const areaResultados = document.getElementById("area-resultados");
            const resAcuracia = document.getElementById("res-acuracia");
            const resAmostras = document.getElementById("res-amostras");
            const matrizCorpo = document.getElementById("matriz-corpo");
            const btnHist = document.getElementById("btn-hist");
            const listaHistorico = document.getElementById("lista-historico");

            form.addEventListener("submit", async (e) => {
                e.preventDefault();
                const ticker = tickerInput.value.trim().toUpperCase();
                if (!ticker) return;

                btnSubmit.disabled = true;
                btnSubmit.textContent = "Auditando...";
                statusRegiao.textContent = "Processando dados do Yahoo e executando TFLite...";

                try {
                    const resp = await fetch("/api/v1/auditar", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ ticker: ticker })
                    });

                    if (!resp.ok) {
                        const erro = await resp.json();
                        throw new Error(erro.detail || "Falha na auditoria.");
                    }

                    const dados = await resp.json();
                    resAcuracia.textContent = (dados.acuracia * 100).toFixed(2) + "%";
                    resAmostras.textContent = dados.total_amostras;

                    matrizCorpo.innerHTML = "";
                    const rotulos = ["Compra", "Venda", "Manter"];
                    dados.matriz_confusao.forEach((linha, i) => {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `<th scope="row">${rotulos[i]}</th>` + linha.map(v => `<td>${v}</td>`).join("");
                        matrizCorpo.appendChild(tr);
                    });

                    areaResultados.classList.remove("hidden");
                    statusRegiao.textContent = "Auditoria finalizada com sucesso.";
                    carregarHistorico();
                } catch (err) {
                    alert(err.message);
                    statusRegiao.textContent = err.message;
                } finally {
                    btnSubmit.disabled = false;
                    btnSubmit.textContent = "Processar";
                }
            });

            async function carregarHistorico() {
                try {
                    const r = await fetch("/api/v1/historico");
                    if (!r.ok) return;
                    const dados = await r.json();
                    listaHistorico.innerHTML = "";
                    dados.forEach(d => {
                        const li = document.createElement("li");
                        const data = new Date(d.executado_em).toLocaleString("pt-BR");
                        li.textContent = `${data} - Ativo: ${d.ticker} | Acurácia: ${(d.acuracia * 100).toFixed(2)}% (${d.amostras_teste} amostras)`;
                        listaHistorico.appendChild(li);
                    });
                } catch (e) {
                    console.error(e);
                }
            }

            btnHist.addEventListener("click", carregarHistorico);
            carregarHistorico();
        });
    </script>
</body>
</html>
"""

# --------------------------------------------------------------------------------------------------
# 5. APLICAÇÃO FASTAPI E ENDPOINTS
# --------------------------------------------------------------------------------------------------
app = FastAPI(
    title="Quant Neural SaaS",
    description="Motor completo de auditoria quantitativa em arquivo único.",
    version="1.0.0"
)

@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
def index():
    return HTMLResponse(content=HTML_EMBUTIDO)

@app.post("/api/v1/auditar", response_model=AuditoriaResposta, tags=["Auditoria"])
def auditar_ativo(req: AuditoriaRequisicao, db: Session = Depends(get_db)):
    ticker = req.ticker.strip().upper()
    try:
        resultado = executar_auditoria_cv2(ticker)
        
        # Gravação no Banco de Dados SQLite
        registro = AuditoriaHistorico(
            ticker=resultado["ticker"],
            acuracia=resultado["acuracia"],
            amostras_teste=resultado["total_amostras"]
        )
        db.add(registro)
        db.commit()
        db.refresh(registro)

        return resultado
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/historico", response_model=List[HistoricoResposta], tags=["Histórico"])
def listar_historico(db: Session = Depends(get_db)):
    return db.query(AuditoriaHistorico).order_by(AuditoriaHistorico.executado_em.desc()).limit(15).all()

# --------------------------------------------------------------------------------------------------
# 6. INICIALIZAÇÃO
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
