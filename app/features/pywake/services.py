import httpx
import xarray as xr
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.path import Path
from pyproj import Transformer
from py_wake.site import XRSite
from py_wake.site.shear import PowerShear
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_farm_models import All2AllIterative
from py_wake.superposition_models import MaxSum
from py_wake.deficit_models import FugaDeficit
from py_wake.flow_map import XYGrid
from py_wake.deficit_models import BastankhahGaussianDeficit
from py_wake.superposition_models import LinearSum
from py_wake.deficit_models import SelfSimilarityDeficit2020

from app.core.settings import settings
from app.features.pywake.schemas import GeoJSONQuery

# ==========================================================
# FUNÇÕES DE CONFIGURAÇÃO
# ==========================================================

async def load_turbine_data(csv_path):
    """Carrega os dados da turbina de um arquivo CSV."""
    try:
        df = pd.read_csv(csv_path)
        ws = df['Wind Speed [m/s]']
        power = df['Power [kW]']
        ct = df['Ct [-]']
        wt = WindTurbine(name='MyWT',
                        diameter=240,
                        hub_height=150,
                        powerCtFunction=PowerCtTabular(ws, power, 'kW', ct))
        print(f"Turbina '{wt.name}' carregada com diâmetro de {wt.diameter()}m.")
        return wt
    except FileNotFoundError:
        print(f"Erro: Arquivo CSV da turbina não encontrado em '{csv_path}'")
        return None

async def setup_simulation_models(freq, c, k, ti, wd, wt):
    """Configura o Site e o Modelo de Simulação do PyWake."""
    ds = xr.Dataset(
        data_vars={
            'Sector_frequency': ('wd', freq),
            'Weibull_A': ('wd', c),
            'Weibull_k': ('wd', k),
            'TI': ('wd', ti)
        },
        coords={'wd': wd}
    )
    
    site = XRSite(ds, shear=PowerShear(h_ref=100, alpha=0.1))
    
    wf_model = All2AllIterative(
        site,
        wt,
        wake_deficitModel=BastankhahGaussianDeficit(use_effective_ws=True),
        superpositionModel=LinearSum(),
        blockage_deficitModel=SelfSimilarityDeficit2020()
    )
    return site, wf_model

# ==========================================================
# FUNÇÕES DE GEOMETRIA E LAYOUT
# ==========================================================

async def load_and_project_boundary(geojson: GeoJSONQuery, source_crs="epsg:4326", target_crs="epsg:32724"):
    """
    Carrega um GeoJSON, extrai coordenadas e as projeta de Lon/Lat para UTM (metros).
    """
    coords_lon_lat = geojson.geometry["coordinates"][0][:-1] # Remove o último ponto (repetido)
    
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    pontos_usuario = []
    for lon, lat in coords_lon_lat:
        x_metro, y_metro = transformer.transform(lon, lat)
        pontos_usuario.append((x_metro, y_metro))
    
    boundary_polygon = np.array(pontos_usuario)
    boundary_path = Path(boundary_polygon)
    
    print(f"Área GeoJSON de {len(pontos_usuario)} pontos carregada e projetada para UTM.")
    return boundary_polygon, boundary_path

async def get_rotation_angle(wind_direction_met):
    """Converte o ângulo do vento (Meteorológico) para o ângulo de rotação (Cartesiano)."""
    angle_deg_cart = (270 - wind_direction_met) % 360
    angle_rad_cart = np.deg2rad(angle_deg_cart)
    print(f"Direção do vento (MET): {wind_direction_met}°, Ângulo de rotação (CART): {angle_deg_cart:.1f}°")
    return angle_rad_cart

async def get_upwind_anchor(boundary_polygon, angle_rad_cart):
    """Encontra o vértice 'upwind' (oposto ao vento/fluxo) da geometria."""
    v_flow = np.array([np.cos(angle_rad_cart), np.sin(angle_rad_cart)])
    dot_products = np.dot(boundary_polygon, v_flow)
    anchor_vertex = boundary_polygon[np.argmin(dot_products)]
    print(f"Ancorando a grade no vértice 'upwind': ({anchor_vertex[0]:.0f}, {anchor_vertex[1]:.0f})")
    return anchor_vertex

async def generate_candidate_grid(boundary_polygon, anchor_vertex, angle_rad_cart, spacing_downwind, spacing_crosswind, stagger_offset):
    """
    Gera uma grade de pontos candidatos, aplicando 'stagger', rotação e ancoragem.
    """
    min_x, min_y = boundary_polygon.min(axis=0)
    max_x, max_y = boundary_polygon.max(axis=0)
    diag = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    
    # Lógica de buffer do seu script (para garantir cobertura)
    side_lengths = np.sqrt(np.sum(np.diff(boundary_polygon, axis=0, append=boundary_polygon[:1])**2, axis=1))
    max_side = np.max(side_lengths)
    extra_pts_downwind = int(max_side // spacing_downwind) + 10
    extra_pts_crosswind = int(max_side // spacing_crosswind) + 10
    n_pts_downwind = int(diag / spacing_downwind) + extra_pts_downwind
    n_pts_crosswind = int(diag / spacing_crosswind) + extra_pts_crosswind

    # Gera a grade base
    x_pts = np.arange(0, n_pts_downwind) * spacing_downwind
    y_pts = np.arange(-n_pts_crosswind // 2, n_pts_crosswind // 2 + 1) * spacing_crosswind
    xx, yy = np.meshgrid(x_pts, y_pts)

    # Aplica 'stagger'
    xx_index = np.round(xx / spacing_downwind).astype(int)
    stagger_mask = (xx_index % 2 == 1)
    yy_staggered = yy + (stagger_mask * stagger_offset)

    # Rotaciona a grade
    cos_a = np.cos(angle_rad_cart)
    sin_a = np.sin(angle_rad_cart)
    x_rot = xx * cos_a - yy_staggered * sin_a
    y_rot = xx * sin_a + yy_staggered * cos_a

    # Move a grade para o ponto âncora
    x_candidatos = x_rot.ravel() + anchor_vertex[0]
    y_candidatos = y_rot.ravel() + anchor_vertex[1]
    
    return np.vstack([x_candidatos, y_candidatos]).T

async def filter_grid_by_boundary(pontos_candidatos, boundary_path):
    """Filtra os pontos candidatos, mantendo apenas os que estão dentro da geometria."""
    mascara = boundary_path.contains_points(pontos_candidatos)
    pontos_finais = pontos_candidatos[mascara]
    x_layout = pontos_finais[:, 0]
    y_layout = pontos_finais[:, 1]
    print(f"Layout gerado com {len(x_layout)} turbinas.")
    return x_layout, y_layout

# ==========================================================
# FUNÇÕES DE SIMULAÇÃO E PLOTAGEM
# ==========================================================

async def run_simulation(polygon: GeoJSONQuery, point_properties: dict, wind_speed=10):
  # --- 1. Configurações Iniciais ---
  
  # Carregar dados da turbina
  wt = await load_turbine_data("./data/IEA_Reference_15MW_240.csv") # Substitua pelo nome do arquivo correto
  if wt is None:
      return # Para a execução se o arquivo não for encontrado

  # Dados do site (do script original)
  freq = np.array(point_properties.get("dp", [])) / 1000
  c = point_properties.get("c_s")
  k = point_properties.get("k_s")
  wd = np.linspace(0, 360, len(freq), endpoint=False)
  ti = [.1] * 12
  
  # --- 2. Setup dos Modelos PyWake ---
  site, wf_model = await setup_simulation_models(freq, c, k, ti, wd, wt)
  
  # --- 3. Processamento da Geometria ---
  
  boundary_polygon, boundary_path = await load_and_project_boundary(polygon)

  # --- 4. Geração do Layout ---
  rotor_diameter = wt.diameter()
  spacing_downwind = 15 * rotor_diameter
  spacing_crosswind = 3 * rotor_diameter
  stagger_offset = 1.5 * rotor_diameter
  
  # Define a direção do vento (METEOROLÓGICO)
  wind_direction_met = wd[np.argmax(freq)] # Para usar o predominante automático
  
  # Gera os pontos candidatos
  angle_rad_cart = await get_rotation_angle(wind_direction_met)
  anchor_vertex = await get_upwind_anchor(boundary_polygon, angle_rad_cart)
  
  pontos_candidatos = await generate_candidate_grid(
      boundary_polygon, anchor_vertex, angle_rad_cart,
      spacing_downwind, spacing_crosswind, stagger_offset
  )
  
  # Filtra os pontos
  x_layout, y_layout = await filter_grid_by_boundary(pontos_candidatos, boundary_path)

  """Roda a simulação do PyWake se houver turbinas."""
  simulacaoResult = wf_model(x_layout, y_layout, wd=wind_direction_met, ws=wind_speed)
  return simulacaoResult


async def generate_geojson(geojson_name: str, polygon: GeoJSONQuery):
    url = f"{settings.MAIN_SERVICE}geojsons/query/centroid/{geojson_name}"

    point_properties = None
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=polygon.model_dump())
        response.raise_for_status()
        point_properties = response.json()

    simulation_result = await run_simulation(polygon, point_properties)
    simulation_result = simulation_result.to_dict()

    transformer = Transformer.from_crs("EPSG:32724", "EPSG:4674", always_xy=True)

    data_vars = simulation_result.get("data_vars", None)
    data_coords = simulation_result.get("coords", None)

    if not data_vars or not data_coords:
        return None

    xs = data_vars["x"]["data"]
    ys = data_vars["y"]["data"]

    lons, lats = transformer.transform(xs, ys)

    geojson_return = {
        "type": "FeatureCollection",
        "features": [],
        "properties": {
            "Weibull_A": data_vars["Weibull_A"]["data"],
            "Weibull_k": data_vars["Weibull_k"]["data"],
            "WS": data_vars["WS"]["data"],
            "WD": data_vars["WD"]["data"]
        }
    }

    for i in range(len(xs)):
        feature = {
            "type": "Feature",
            "properties": {
                "wt": i + 1,
                "Aerogerador": "Modelo Catatau 15MW",
                "Pot": 15.0,
                "WS_eff": float(data_vars["WS_eff"]["data"][i][0][0]),
                "TI_eff": float(data_vars["TI_eff"]["data"][i][0][0]),
                "Power": float(data_vars["Power"]["data"][i][0][0]),
                "CT": float(data_vars["CT"]["data"][i][0][0]),
                "h": float(data_vars["h"]["data"][i])
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lons[i], lats[i]]
            }
        }

        geojson_return["features"].append(feature)

    return geojson_return
