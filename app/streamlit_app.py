"""
Real-time fraud monitoring dashboard / Дашборд мониторинга fraud в реальном времени.

Streams pre-scored transactions (model output from a submission file joined with
the raw transaction fields) as if they arrive live. When a transaction's fraud
probability crosses the threshold, a toast alert fires and the transaction is
routed to the "fraud control" review queue. Live statistics update on every tick.

Проигрывает заранее отскоренные транзакции (предсказания модели из файла submission,
объединённые с сырыми полями транзакций), как будто они приходят в реальном времени.
Когда вероятность fraud превышает порог — всплывает toast-уведомление, а транзакция
отправляется в очередь "отдела контроля". Статистика обновляется на каждом такте.

Итоговая модель проекта — настроенный LightGBM (`submissions/submission_lgb.csv`),
но в сайдбаре можно выбрать любой сабмишен из папки `submissions/`.
The project's final model is the tuned LightGBM (`submissions/submission_lgb.csv`),
but any submission file in `submissions/` can be picked in the sidebar.

Run / Запуск:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Paths / Пути -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
TEST_TX_PATH = ROOT_DIR / "data" / "test_transaction.csv"

# Final model of the project / Итоговая модель проекта
DEFAULT_SUBMISSION = "submission_lgb.csv"

# Raw fields we surface in the UI / Сырые поля, которые показываем в интерфейсе
DISPLAY_COLS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card4",      # payment network / платёжная система
    "card6",      # card type / тип карты
    "P_emaildomain",
    "addr1",
]

# How many transactions to load for the demo stream / Сколько грузим для демо-потока
MAX_LOAD = 10000


# --- Data loading / Загрузка данных -----------------------------------------
def list_submissions() -> list[str]:
    """Available submission files / Доступные файлы сабмишенов."""
    files = sorted(p.name for p in SUBMISSIONS_DIR.glob("submission*.csv"))
    return files or [DEFAULT_SUBMISSION]


@st.cache_data(show_spinner="Загрузка транзакций… / Loading transactions…")
def load_stream(submission_name: str, max_rows: int = MAX_LOAD) -> pd.DataFrame:
    """
    Join model scores with raw transaction fields, ordered by time.
    Объединяет предсказания модели с сырыми полями транзакций, по времени.
    """
    scores = pd.read_csv(SUBMISSIONS_DIR / submission_name)  # TransactionID, isFraud
    scores = scores.rename(columns={"isFraud": "fraud_proba"})

    tx = pd.read_csv(TEST_TX_PATH, usecols=DISPLAY_COLS, nrows=max_rows)

    df = tx.merge(scores, on="TransactionID", how="inner")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    return df


# --- Session state / Состояние сессии ---------------------------------------
def init_state() -> None:
    """Initialize per-session counters / Инициализация счётчиков сессии."""
    defaults = {
        "running": False,
        "cursor": 0,
        "processed": 0,
        "flagged": 0,
        "frozen_amt": 0.0,
        "proba_sum": 0.0,
        "recent": deque(maxlen=12),      # last transactions / последние транзакции
        "proba_hist": deque(maxlen=60),  # probability sparkline / лента вероятностей
        "queue": [],                     # review queue / очередь контроля
        "alert": None,                   # current alert card / текущая карточка алерта
        "alert_ttl": 0,                  # ticks the card stays visible / сколько тактов висит
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_run() -> None:
    """Reset the stream / Сброс потока."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


# --- UI ---------------------------------------------------------------------
st.set_page_config(page_title="Fraud Monitoring", page_icon="🚨", layout="wide")
init_state()

st.title("🚨 Real-Time Transaction Monitoring · Мониторинг транзакций")
st.caption(
    "Stream of pre-scored transactions (tuned LightGBM). High fraud probability "
    "triggers an alert and routing to the control desk. · "
    "Поток заранее отскоренных транзакций (настроенный LightGBM): при высокой "
    "вероятности fraud — алерт и отправка в отдел контроля."
)

# --- Sidebar controls / Панель управления -----------------------------------
with st.sidebar:
    st.header("⚙️ Управление")
    submissions = list_submissions()
    default_idx = submissions.index(DEFAULT_SUBMISSION) if DEFAULT_SUBMISSION in submissions else 0
    submission_name = st.selectbox(
        "Источник скора · Score source", submissions, index=default_idx,
        help="Файл с вероятностями из ноутбука 03. По умолчанию — итоговый LightGBM.",
    )
    threshold = st.slider(
        "Порог fraud (P)", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Транзакции с вероятностью ≥ порога уходят в отдел контроля.",
    )
    speed = st.slider(
        "Задержка между тактами, сек", min_value=0.1, max_value=2.0, value=0.6, step=0.1,
    )
    batch = st.slider(
        "Транзакций за такт", min_value=1, max_value=10, value=1, step=1,
    )

    c1, c2 = st.columns(2)
    if c1.button("▶️ Старт", use_container_width=True, type="primary"):
        st.session_state.running = True
    if c2.button("⏸️ Пауза", use_container_width=True):
        st.session_state.running = False
    if st.button("🔄 Сброс", use_container_width=True):
        reset_run()
        st.rerun()

df = load_stream(submission_name)
n_total = len(df)

with st.sidebar:
    st.divider()
    st.metric("Total in stream · Всего в потоке", f"{n_total:,}")
    st.progress(min(st.session_state.cursor / n_total, 1.0) if n_total else 0.0)


# --- Stream step / Шаг потока -----------------------------------------------
# Process the next batch before rendering, so the UI reflects fresh state.
# Обрабатываем следующую пачку до рендера, чтобы UI показывал свежее состояние.
if st.session_state.running and st.session_state.cursor < n_total:
    end = min(st.session_state.cursor + batch, n_total)
    chunk = df.iloc[st.session_state.cursor:end]

    for _, row in chunk.iterrows():
        proba = float(row["fraud_proba"])
        is_fraud = proba >= threshold

        st.session_state.processed += 1
        st.session_state.proba_sum += proba
        st.session_state.proba_hist.append(proba)
        st.session_state.recent.appendleft(
            {
                "ID": int(row["TransactionID"]),
                "Сумма": float(row["TransactionAmt"]),
                "Карта": f"{row['card4']}/{row['card6']}",
                "Email": row["P_emaildomain"],
                "P(fraud)": round(proba, 3),
                "Флаг": "🚩" if is_fraud else "✅",
            }
        )

        if is_fraud:
            st.session_state.flagged += 1
            st.session_state.frozen_amt += float(row["TransactionAmt"])
            alert = {
                "ID": int(row["TransactionID"]),
                "Сумма": float(row["TransactionAmt"]),
                "Карта": f"{row['card4']}/{row['card6']}",
                "Email": row["P_emaildomain"],
                "Регион": row["addr1"],
                "Продукт": row["ProductCD"],
                "P": round(proba, 3),
            }
            st.session_state.queue.insert(0, {**alert, "Статус": "🚩 на проверке"})
            st.session_state.alert = alert
            st.session_state.alert_ttl = 4  # держим карточку ~4 такта / keep ~4 ticks
            st.toast(
                f"🚩 FRAUD ALERT · ID {alert['ID']} · "
                f"${alert['Сумма']:,.2f} · P={alert['P']} → отдел контроля",
                icon="🚨",
            )

    st.session_state.cursor = end


# --- Metrics / Метрики ------------------------------------------------------
processed = st.session_state.processed
flagged = st.session_state.flagged
avg_p = st.session_state.proba_sum / processed if processed else 0.0
fraud_rate = flagged / processed * 100 if processed else 0.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Processed · Обработано", f"{processed:,}")
m2.metric("Flagged 🚩 · Помечено", f"{flagged:,}")
m3.metric("Fraud rate · Доля", f"{fraud_rate:.2f}%")
m4.metric("Frozen $ · Заморожено", f"{st.session_state.frozen_amt:,.0f}")
m5.metric("Avg P · Средняя", f"{avg_p:.3f}")

# --- Alert card / Карточка алерта -------------------------------------------
alert_box = st.empty()
if st.session_state.alert_ttl > 0 and st.session_state.alert:
    a = st.session_state.alert
    alert_box.error(
        f"### 🚩 Suspicious transaction → control desk · Подозрительная операция\n"
        f"**ID** {a['ID']}  ·  **Amount/Сумма** ${a['Сумма']:,.2f}  ·  "
        f"**Card/Карта** {a['Карта']}  ·  **Email** {a['Email']}  ·  "
        f"**Region/Регион** {a['Регион']}  ·  **Product/Продукт** {a['Продукт']}  ·  "
        f"**P(fraud)** {a['P']}"
    )
    st.session_state.alert_ttl -= 1
else:
    alert_box.empty()

# --- Probability stream + recent transactions / Поток и последние транзакции -
left, right = st.columns([3, 2])

with left:
    st.subheader("📈 Probability stream · Поток вероятностей")
    if st.session_state.proba_hist:
        hist = list(st.session_state.proba_hist)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=hist, mode="lines+markers", name="P(fraud)",
                line=dict(color="#1f77b4"),
                marker=dict(
                    color=["#d62728" if p >= threshold else "#1f77b4" for p in hist],
                    size=6,
                ),
            )
        )
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="#d62728",
            annotation_text=f"threshold {threshold}",
        )
        fig.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_range=[0, 1], showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Press ▶️ Start to launch the stream. · Нажмите ▶️ Старт.")

with right:
    st.subheader("🧾 Recent transactions · Последние транзакции")
    if st.session_state.recent:
        st.dataframe(
            pd.DataFrame(list(st.session_state.recent)),
            use_container_width=True, hide_index=True, height=300,
        )
    else:
        st.info("—")

# --- Review queue (full width, bottom) / Очередь контроля (снизу, на всю ширину)
st.divider()
st.subheader(f"🛡️ Fraud Control Queue · Отдел контроля ({len(st.session_state.queue)})")
if st.session_state.queue:
    queue_df = pd.DataFrame(st.session_state.queue)
    st.dataframe(queue_df, use_container_width=True, hide_index=True, height=300)
    st.download_button(
        "⬇️ Export queue (CSV) · Выгрузить очередь",
        queue_df.to_csv(index=False).encode("utf-8"),
        file_name="fraud_review_queue.csv",
        mime="text/csv",
    )
else:
    st.info("No suspicious transactions yet. · Пока нет подозрительных операций.")


# --- Drive the stream / Двигаем поток ---------------------------------------
# Self-rerun loop: keep streaming while running and data remains.
# Цикл саморендера: продолжаем поток, пока запущено и есть данные.
if st.session_state.running and st.session_state.cursor < n_total:
    time.sleep(speed)
    st.rerun()
elif st.session_state.cursor >= n_total and processed:
    st.session_state.running = False
    st.success("✅ Stream finished. · Поток завершён.")
