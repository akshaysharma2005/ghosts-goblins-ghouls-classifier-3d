import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from .data_loader import CREATURE_COLORS

def build_2d_biometric_violins(df, metric="bone_length"):
    """Comparative violin distribution by creature type."""
    fig = px.violin(
        df,
        x="type",
        y=metric,
        color="type",
        box=True,
        points="all",
        color_discrete_map=CREATURE_COLORS,
        labels={"type": "Creature Species", metric: metric.replace("_", " ").title()}
    )
    fig.update_layout(
        title=f"Biometric Distribution: {metric.replace('_', ' ').title()}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )
    return fig

def build_color_composition(df):
    """Creature species breakdown across spectral aura colors."""
    pivot = df.groupby(["color", "type"]).size().unstack(fill_value=0).reset_index()
    
    fig = go.Figure()
    for creature in ["Ghoul", "Goblin", "Ghost"]:
        fig.add_trace(go.Bar(
            x=pivot["color"],
            y=pivot[creature],
            name=creature,
            marker_color=CREATURE_COLORS[creature]
        ))
        
    fig.update_layout(
        title="Spectral Aura Color Frequency by Species",
        barmode="group",
        xaxis_title="Observed Aura Color",
        yaxis_title="Specimen Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=380,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def build_3d_morphospace(df, show_hulls=True):
    """
    3D Supernatural Biometric Morphospace:
    X: Bone Length
    Y: Rotting Flesh
    Z: Hair Length
    Color: Creature Type
    Size: Has Soul
    """
    fig = go.Figure()

    for creature, color in CREATURE_COLORS.items():
        sub = df[df["type"] == creature]
        fig.add_trace(go.Scatter3d(
            x=sub["bone_length"],
            y=sub["rotting_flesh"],
            z=sub["hair_length"],
            mode="markers",
            name=creature,
            marker=dict(
                size=sub["has_soul"] * 10 + 3,
                color=color,
                opacity=0.85,
                line=dict(color="rgba(255,255,255,0.4)", width=1)
            ),
            customdata=np.stack((sub["id"], sub["color"], sub["has_soul"].round(3)), axis=-1),
            hovertemplate="<b>%{data.name} (ID: %{customdata[0]})</b><br>Bone Length: %{x:.2f}<br>Rotting Flesh: %{y:.2f}<br>Hair Length: %{z:.2f}<br>Soul Power: %{customdata[2]}<br>Color: %{customdata[1]}<extra></extra>"
        ))

        # Add Convex Hull Enclosure
        if show_hulls:
            pts = sub[["bone_length", "rotting_flesh", "hair_length"]].values
            if len(pts) >= 4:
                try:
                    hull = ConvexHull(pts)
                    fig.add_trace(go.Mesh3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                        color=color,
                        opacity=0.12,
                        name=f"{creature} Convex Envelope",
                        showlegend=False
                    ))
                except Exception:
                    pass

    fig.update_layout(
        title="3D Biometric Morphospace: Bone Length × Rotting Flesh × Hair Length (Size: Soul Power)",
        scene=dict(
            xaxis=dict(title="Bone Length", backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Rotting Flesh", backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Hair Length", backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=620,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

def build_4d_soul_resonance_animation(df):
    """
    4D Spatio-Temporal Animation:
    - Dim 1 (X): Bone Length
    - Dim 2 (Y): Rotting Flesh
    - Dim 3 (Z): Hair Length
    - Dim 4 (Color): Creature Type
    - Dim 5 (Size): Has Soul
    - Animation Frame: Soul Resonance Tier (Aura progression from Faint to Transcendent)
    """
    df_anim = df.sort_values(by="has_soul").copy()

    fig = px.scatter_3d(
        df_anim,
        x="bone_length",
        y="rotting_flesh",
        z="hair_length",
        color="type",
        size="has_soul",
        size_max=18,
        animation_frame="soul_tier_str",
        animation_group="id",
        hover_name="type",
        hover_data={"color": True, "has_soul": ":.3f"},
        color_discrete_map=CREATURE_COLORS,
        range_x=[0, 1.05],
        range_y=[0, 1.05],
        range_z=[0, 1.05],
        labels={
            "bone_length": "Bone Length",
            "rotting_flesh": "Rotting Flesh",
            "hair_length": "Hair Length"
        }
    )

    fig.update_layout(
        title="4D Soul Resonance Evolution (Watch Spectral Separation Across Soul Luminosity)",
        scene=dict(
            xaxis=dict(title="Bone Length", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Rotting Flesh", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Hair Length", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=620,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig
