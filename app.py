"""
dashboard/app.py — Streamlit Visualisation Dashboard
=====================================================
Interactive frontend for the Cloud Workload Scheduler.
Fetches scheduling results from the FastAPI backend and renders
comparative visualisations for GA and PSO.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cloud Workload Scheduler",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1B2A4A 0%, #2E5DA8 100%);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }
    .winner-badge {
        background: #27AE60; border-radius: 8px; padding: 6px 14px;
        color: white; font-weight: 700; display: inline-block;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
             use_column_width=True, caption="")
    st.title("☁️ Scheduler Config")
    st.markdown("---")
    api_url   = st.text_input("API URL", value="http://127.0.0.1:8000")
    num_vms   = st.slider("Virtual Machines", min_value=2, max_value=10, value=5, step=1)
    batch_sz  = st.slider("Batch Size (tasks)", min_value=10, max_value=500, value=100, step=10)
    dataset   = st.selectbox("Dataset Mode", ["planetlab", "live"])
    run_btn   = st.button("🚀 Run Scheduler", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**Team**\n- Kimaya Mishra\n- Naveen Singh\n- M. Nabil Khan\n- M. Musab")
    st.markdown("*PSIT Kanpur | AKTU 2026*")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("☁️ Cloud Workload Scheduler")
st.markdown(
    "**ML-powered dynamic task scheduling** using *Genetic Algorithm* and "
    "*Particle Swarm Optimization*, with RandomForest demand prediction."
)
st.markdown("---")

# ── Fetch + display ───────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running ML prediction + GA + PSO..."):
        try:
            resp = requests.get(
                f"{api_url}/schedule",
                params={"num_vms": num_vms, "batch_size": batch_sz, "dataset": dataset},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state["result"] = data
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the backend. Run `uvicorn backend.main:app --reload` first.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

if "result" not in st.session_state:
    st.info("👈 Configure parameters in the sidebar and click **Run Scheduler** to begin.")
    st.stop()

data = st.session_state["result"]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📦 Tasks",       data["num_tasks"])
col2.metric("🖥️ VMs",         data["num_vms"])
col3.metric("🧬 GA Makespan", f"{data['ga_makespan']:.4f}")
col4.metric("🐦 PSO Makespan",f"{data['pso_makespan']:.4f}")
col5.metric("🏆 Winner",      data["winner"])

st.markdown("---")

# ── VM Load Distribution Charts ───────────────────────────────────────────────
vm_labels = [f"VM-{i}" for i in range(data["num_vms"])]
c1, c2 = st.columns(2)

with c1:
    st.subheader("🧬 GA — VM Load Distribution")
    fig_ga = go.Figure(go.Bar(
        x=vm_labels, y=data["vm_loads_ga"],
        marker_color=["#1B2A4A" if v < max(data["vm_loads_ga"]) else "#E74C3C"
                      for v in data["vm_loads_ga"]],
        text=[f"{v:.3f}" for v in data["vm_loads_ga"]],
        textposition="outside",
    ))
    fig_ga.add_hline(y=sum(data["vm_loads_ga"])/len(data["vm_loads_ga"]),
                     line_dash="dash", line_color="#F39C12", annotation_text="Mean")
    fig_ga.update_layout(
        yaxis_title="CPU Load", template="plotly_white", height=320,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_ga, use_container_width=True)

with c2:
    st.subheader("🐦 PSO — VM Load Distribution")
    fig_pso = go.Figure(go.Bar(
        x=vm_labels, y=data["vm_loads_pso"],
        marker_color=["#1C7293" if v < max(data["vm_loads_pso"]) else "#E74C3C"
                      for v in data["vm_loads_pso"]],
        text=[f"{v:.3f}" for v in data["vm_loads_pso"]],
        textposition="outside",
    ))
    fig_pso.add_hline(y=sum(data["vm_loads_pso"])/len(data["vm_loads_pso"]),
                      line_dash="dash", line_color="#F39C12", annotation_text="Mean")
    fig_pso.update_layout(
        yaxis_title="CPU Load", template="plotly_white", height=320,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_pso, use_container_width=True)

# ── Makespan Comparison ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Algorithm Comparison")
c1, c2 = st.columns([2, 1])

with c1:
    fig_cmp = go.Figure()
    fig_cmp.add_bar(name="GA",  x=["Makespan"], y=[data["ga_makespan"]],
                    marker_color="#1B2A4A",
                    text=[f"{data['ga_makespan']:.4f}"], textposition="outside")
    fig_cmp.add_bar(name="PSO", x=["Makespan"], y=[data["pso_makespan"]],
                    marker_color="#1C7293",
                    text=[f"{data['pso_makespan']:.4f}"], textposition="outside")
    fig_cmp.update_layout(
        barmode="group", template="plotly_white", height=300,
        yaxis_title="Makespan", margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

with c2:
    st.markdown("### Results Summary")
    st.markdown(f"| Metric | GA | PSO |")
    st.markdown(f"|---|---|---|")
    st.markdown(f"| Makespan | **{data['ga_makespan']:.4f}** | {data['pso_makespan']:.4f} |")
    st.markdown(f"| Exec Time (s) | {data['exec_time_ga']} | {data['exec_time_pso']} |")
    st.markdown(f"| Winner | {'✅' if data['winner']=='GA' else '  '} | {'✅' if data['winner']=='PSO' else '  '} |")
    st.markdown(f"\n**Improvement:** `{data['improvement_pct']}%`")

# ── Task Assignment Table ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Task Assignment Details")
df_tasks = pd.DataFrame({
    "Task ID":      list(range(data["num_tasks"])),
    "CPU Demand":   [f"{c:.4f}" for c in data["cpu_loads"]],
    "GA → VM":      [f"VM-{v}" for v in data["ga_schedule"]],
    "PSO → VM":     [f"VM-{v}" for v in data["pso_schedule"]],
    "Same VM?":     ["✅" if a == b else "❌"
                     for a, b in zip(data["ga_schedule"], data["pso_schedule"])],
})
st.dataframe(df_tasks, use_container_width=True, height=300)
