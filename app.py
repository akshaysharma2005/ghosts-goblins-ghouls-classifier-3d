import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import load_monster_data, train_voting_classifier, CREATURE_COLORS
from src.visualizer import (
    build_2d_biometric_violins,
    build_color_composition,
    build_3d_morphospace,
    build_4d_soul_resonance_animation
)

st.set_page_config(
    page_title="Supernatural Creature 4D Classifier & Biometric Space",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom gothic dark sci-fi styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .creature-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_monster_data("data/monsters.csv")

df = get_data()

@st.cache_resource
def get_classifier():
    return train_voting_classifier(df)

model_pipeline, sub_models = get_classifier()

# Sidebar
st.sidebar.title("👻 Spectral Biometry")
st.sidebar.markdown("Cryptid Biometric Morphospaces & Ensemble Voting Classification AI.")

species_filter = st.sidebar.multiselect("Filter Species", options=["Ghoul", "Goblin", "Ghost"], default=["Ghoul", "Goblin", "Ghost"])
color_filter = st.sidebar.multiselect("Spectral Aura Color", options=sorted(df["color"].unique()), default=sorted(df["color"].unique()))

filtered_df = df[(df["type"].isin(species_filter)) & (df["color"].isin(color_filter))]

# Header
col_title, col_kpis = st.columns([2, 3])
with col_title:
    st.title("Supernatural 4D Biometric Classifier")
    st.caption("3D morphospace clustering, 4D soul resonance animation, and multi-model ensemble voting classification.")

with col_kpis:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="creature-card">
            <span style="color: #94a3b8; font-size: 11px;">SPECIMENS</span>
            <h3 style="margin: 0; color: #38bdf8;">{len(filtered_df)}</h3>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="creature-card">
            <span style="color: #94a3b8; font-size: 11px;">GHOULS</span>
            <h3 style="margin: 0; color: #10b981;">{(filtered_df['type'] == 'Ghoul').sum()}</h3>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="creature-card">
            <span style="color: #94a3b8; font-size: 11px;">GOBLINS</span>
            <h3 style="margin: 0; color: #f59e0b;">{(filtered_df['type'] == 'Goblin').sum()}</h3>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="creature-card">
            <span style="color: #94a3b8; font-size: 11px;">GHOSTS</span>
            <h3 style="margin: 0; color: #38bdf8;">{(filtered_df['type'] == 'Ghost').sum()}</h3>
        </div>
        """, unsafe_allow_html=True)

# Tabs
tab_2d, tab_3d, tab_4d, tab_ml = st.tabs([
    "📊 2D Biometric Profiling",
    "🌐 3D Biometric Morphospace",
    "⏳ 4D Soul Resonance Evolution",
    "🔮 Live Creature Voting Classifier"
])

# TAB 1: 2D
with tab_2d:
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        metric_choice = st.selectbox("Select Biometric Dimension", ["bone_length", "rotting_flesh", "hair_length", "has_soul"])
        st.plotly_chart(build_2d_biometric_violins(filtered_df, metric_choice), use_container_width=True)
    with col_v2:
        st.plotly_chart(build_color_composition(filtered_df), use_container_width=True)

# TAB 2: 3D
with tab_3d:
    st.subheader("3D Biometric Morphospace & Species Convex Envelopes")
    st.markdown("Cryptid coordinates plotted in 3D: Bone Length × Rotting Flesh × Hair Length. Marker sizes indicate Soul Luminosity.")
    show_mesh = st.checkbox("Display Species 3D Convex Hull Enclosures", value=True)
    st.plotly_chart(build_3d_morphospace(filtered_df, show_hulls=show_mesh), use_container_width=True)

# TAB 3: 4D
with tab_4d:
    st.subheader("4D Soul Resonance Evolution (Hyper-Dimensional Projection)")
    st.markdown("Press **Play** to observe how creature morphospaces segregate across escalating tiers of Soul Power (Faint Aura to Transcendent).")
    st.plotly_chart(build_4d_soul_resonance_animation(filtered_df), use_container_width=True)

# TAB 4: ML
with tab_ml:
    st.subheader("🔮 Cryptid Biometric Identification & Ensemble Voting Engine")
    st.markdown("Input physical measurements of an unidentified creature specimen to run ensemble consensus classification.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        in_bone = st.slider("Bone Length Index", 0.0, 1.0, 0.52, 0.01)
        in_flesh = st.slider("Rotting Flesh Index", 0.0, 1.0, 0.44, 0.01)
        in_hair = st.slider("Hair Length Index", 0.0, 1.0, 0.65, 0.01)
    with col_c2:
        in_soul = st.slider("Soul Power / Luminosity", 0.0, 1.0, 0.75, 0.01)
        in_color = st.selectbox("Observed Aura Color", sorted(df["color"].unique()))
        
    specimen = pd.DataFrame([{
        "bone_length": in_bone,
        "rotting_flesh": in_flesh,
        "hair_length": in_hair,
        "has_soul": in_soul,
        "color": in_color
    }])
    
    pred_species = model_pipeline.predict(specimen)[0]
    probs = model_pipeline.predict_proba(specimen)[0]
    classes = model_pipeline.classes_
    
    st.markdown("---")
    st.markdown("### Ensemble Consensus Verdict")
    
    res_l, res_r = st.columns([1, 2])
    with res_l:
        badge_color = CREATURE_COLORS.get(pred_species, "#38bdf8")
        st.markdown(f"""
        <div class="creature-card" style="text-align: center; border-left: 6px solid {badge_color};">
            <span style="color: #94a3b8; font-size: 12px;">CLASSIFIED TAXONOMY</span>
            <h1 style="margin: 8px 0; color: {badge_color};">{pred_species}</h1>
            <span style="font-size: 14px; color: #f8fafc;">Confidence: <b>{max(probs) * 100:.1f}%</b></span>
        </div>
        """, unsafe_allow_html=True)
        
    with res_r:
        st.markdown("#### Model Probability Breakdown")
        for cls_name, prob in zip(classes, probs):
            st.write(f"**{cls_name}**: `{prob * 100:.1f}%`")
            st.progress(float(prob))
