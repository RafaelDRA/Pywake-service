import httpx
import xarray as xr
import numpy as np
import pandas as pd
import json
import base64
import binascii
import matplotlib.pyplot as plt
from matplotlib.path import Path
from pyproj import Transformer
from py_wake.site import XRSite
from py_wake.site.shear import PowerShear
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_farm_models import All2AllIterative
from py_wake.deficit_models import FugaDeficit
from py_wake.superposition_models import LinearSum
from shapely.geometry import Polygon
from fastapi import HTTPException, status
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

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
        wake_deficitModel=FugaDeficit(),
        superpositionModel=LinearSum(),
        blockage_deficitModel=FugaDeficit()
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
    
    return boundary_polygon, boundary_path

async def get_rotation_angle(wind_direction_met):
    """Converte o ângulo do vento (Meteorológico) para o ângulo de rotação (Cartesiano)."""
    angle_deg_cart = (270 - wind_direction_met) % 360
    angle_rad_cart = np.deg2rad(angle_deg_cart)
    return angle_rad_cart

async def get_upwind_anchor(boundary_polygon, angle_rad_cart):
    """Encontra o vértice 'upwind' (oposto ao vento/fluxo) da geometria."""
    v_flow = np.array([np.cos(angle_rad_cart), np.sin(angle_rad_cart)])
    dot_products = np.dot(boundary_polygon, v_flow)
    anchor_vertex = boundary_polygon[np.argmin(dot_products)]
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
    return x_layout, y_layout


def _as_sector_array(value, n_sectors, fallback):
    arr = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    if arr.size == 0:
        return np.full(n_sectors, fallback, dtype=float)
    if arr.size == 1:
        return np.full(n_sectors, float(arr[0]), dtype=float)
    if arr.size != n_sectors:
        src = np.linspace(0.0, 1.0, arr.size)
        dst = np.linspace(0.0, 1.0, n_sectors)
        arr = np.interp(dst, src, arr)
    return arr


def _extract_scalar(data, idx, default=0.0):
    arr = np.asarray(data)
    if arr.size == 0:
        return float(default)
    if arr.ndim == 0:
        return float(arr)
    try:
        selector = (idx,) + (0,) * (arr.ndim - 1)
        return float(arr[selector])
    except Exception:
        flat = arr.reshape(-1)
        safe_idx = min(idx, flat.size - 1)
        return float(flat[safe_idx])


_POWER_CURVE_CACHE = None


def _get_power_curve():
    global _POWER_CURVE_CACHE
    if _POWER_CURVE_CACHE is not None:
        return _POWER_CURVE_CACHE

    df = pd.read_csv("./data/IEA_Reference_15MW_240.csv")
    ws_curve = df['Wind Speed [m/s]'].to_numpy(dtype=float)
    power_curve_w = df['Power [kW]'].to_numpy(dtype=float) * 1e3
    _POWER_CURVE_CACHE = (ws_curve, power_curve_w)
    return _POWER_CURVE_CACHE


def _estimate_no_wake_power_w(ws_value):
    ws_curve, power_curve_w = _get_power_curve()
    ws_safe = max(float(ws_value), 0.0)
    return float(np.interp(ws_safe, ws_curve, power_curve_w, left=0.0, right=power_curve_w[-1]))

# ==========================================================
# FUNÇÕES DE SIMULAÇÃO E PLOTAGEM
# ==========================================================

async def run_simulation(polygon: GeoJSONQuery, point_properties: dict, wind_speed: float, all_data=False):
  # --- 1. Configurações Iniciais ---
  
  # Carregar dados da turbina
  wt = await load_turbine_data("./data/IEA_Reference_15MW_240.csv") # Substitua pelo nome do arquivo correto
  if wt is None:
      return # Para a execução se o arquivo não for encontrado

  # Dados do site (do script original)
  # dp values are in permille (‰), sum to 1000, need to convert to fractions (0-1)
  # BUT: Check if already normalized - if sum < 10, assume already in decimal format
  dp_raw = np.asarray(point_properties.get("dp", []), dtype=float).reshape(-1)
  dp_raw = np.nan_to_num(dp_raw, nan=0.0, posinf=0.0, neginf=0.0)

  if dp_raw.size == 0:
      dp_raw = np.ones(12, dtype=float)
  elif dp_raw.size == 1:
      # Single-sector values are expanded to avoid degenerate site setup
      dp_raw = np.full(12, max(float(dp_raw[0]), 1.0), dtype=float)

  dp_sum = dp_raw.sum()
  
  if dp_sum > 10:  # Likely in permille (sum ~1000) or percentage (sum ~100)
      freq = dp_raw / dp_sum  # Normalize to sum to 1.0
  else:  # Already in decimal format
      freq = dp_raw / dp_raw.sum() if dp_raw.sum() > 0 else dp_raw
  n_sectors = len(freq)
  c = _as_sector_array(point_properties.get("c_s"), n_sectors, fallback=10.0)
  k = _as_sector_array(point_properties.get("k_s"), n_sectors, fallback=2.0)
  c = np.clip(np.nan_to_num(c, nan=10.0, posinf=10.0, neginf=10.0), 1e-3, None)
  k = np.clip(np.nan_to_num(k, nan=2.0, posinf=2.0, neginf=2.0), 1e-3, None)
  wd = np.linspace(0, 360, len(freq), endpoint=False)
  ti = [0.1] * len(freq)
  
  # --- 2. Setup dos Modelos PyWake ---
  site, wf_model = await setup_simulation_models(freq, c, k, ti, wd, wt)
  
  # --- 3. Processamento da Geometria ---
  
  boundary_polygon, boundary_path = await load_and_project_boundary(polygon)

  if Polygon(boundary_polygon).area < 2000000:
      raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Simulation area is too small. Please select a larger area.")

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

  if len(x_layout) == 0:
      raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No turbines could be placed in the selected area. Please select a larger or different area.")

  """Roda a simulação do PyWake se houver turbinas."""
  if all_data:
    simulacaoResult = wf_model(x_layout, y_layout, wd=wd, ws=range(3,26))
  else:
    simulacaoResult = wf_model(x_layout, y_layout, wd=wind_direction_met, ws=wind_speed)

  return simulacaoResult


async def generate_geojson(geojson_name: str, polygon: GeoJSONQuery):
    try:
        point_properties = await get_point_from_service(geojson_name, polygon)

        if not point_properties:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Could not retrieve point properties for {geojson_name}"
            )

        # Run simulation for all conditions (AEP)
        simulation_result_all = await run_simulation(polygon=polygon, point_properties=point_properties, wind_speed=point_properties.get("mys", 10), all_data=True)
        
        if simulation_result_all is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Simulation failed. Check turbine data or geometry."
            )

        aep = simulation_result_all.aep()
        aep_por_wt_ws = aep.sum(['wt', 'ws'])  # AEP vs Wind Direction
        aep_por_wt_wd = aep.sum(['wt','wd'])   # AEP vs Wind Speed

        aep_vs_ws_gwh = aep_por_wt_wd.values.tolist()
        aep_vs_wd_gwh = aep_por_wt_ws.values.tolist()

        wind_speed_x = aep_por_wt_wd.ws.values.tolist()
        wind_direction_x = aep_por_wt_ws.wd.values.tolist()
        
        all_data_area_stats = {
            "wind_speed_x": wind_speed_x,
            "aep_vs_ws_y": aep_vs_ws_gwh,
            "wind_direction_x": wind_direction_x,
            "aep_vs_wd_y": aep_vs_wd_gwh
        }

        # Wake by direction
        wake_direction_x = []
        wake_intensity_pct_y = []
        wake_accumulated_gwh_y = []
        try:
            ws_all = simulation_result_all.WS
            ws_eff_all = simulation_result_all.WS_eff
            wake_ratio = xr.where(ws_all > 0, (ws_all - ws_eff_all) / ws_all, 0.0)
            wake_ratio = xr.where(wake_ratio > 0, wake_ratio, 0.0)
            wake_ratio = xr.where(wake_ratio < 1.0, wake_ratio, 1.0)
            reduce_dims = [dim for dim in wake_ratio.dims if dim != 'wd']
            wake_by_wd_pct = wake_ratio.mean(dim=reduce_dims) * 100.0 if reduce_dims else wake_ratio * 100.0

            # Align with meteorological "from" convention used in production wind rose
            wake_direction_x_raw = wake_by_wd_pct.wd.values.tolist() if 'wd' in wake_by_wd_pct.coords else wind_direction_x
            wake_direction_x = [float(wd) for wd in wake_direction_x_raw]
            wake_intensity_pct_y = np.nan_to_num(wake_by_wd_pct.values, nan=0.0, posinf=0.0, neginf=0.0).tolist()

            # Accumulated wake-loss proxy by direction (GWh): directional production * directional wake intensity
            # This emphasizes total directional impact over mean per-WTG topology effects.
            wake_ratio_decimal = np.array(wake_intensity_pct_y, dtype=float) / 100.0
            directional_production_gwh = np.array(aep_vs_wd_gwh, dtype=float)
            wake_accumulated_gwh_y = (directional_production_gwh * wake_ratio_decimal).tolist()
        except Exception:
            wake_direction_x = wind_direction_x
            wake_intensity_pct_y = [0.0 for _ in wake_direction_x]
            wake_accumulated_gwh_y = [0.0 for _ in wake_direction_x]

        all_data_area_stats.update({
            "wake_direction_x": wake_direction_x,
            "wake_intensity_pct_y": wake_intensity_pct_y,
            "wake_accumulated_gwh_y": wake_accumulated_gwh_y
        })

        farm_aep = simulation_result_all.aep().sum().item()

        simulation_result_one_d = await run_simulation(polygon=polygon, point_properties=point_properties, wind_speed=point_properties.get("mys", 10))
    
        if simulation_result_one_d is None:
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Simulation failed for single condition."
            )

        simulation_result_one_d = simulation_result_one_d.to_dict()
        transformer = Transformer.from_crs("EPSG:32724", "EPSG:4674", always_xy=True)

        data_vars = simulation_result_one_d.get("data_vars", None)
        data_coords = simulation_result_one_d.get("coords", None)

        if not data_vars or not data_coords:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Simulation result is empty or invalid."
            )

        xs = data_vars["x"]["data"]
        ys = data_vars["y"]["data"]

        lons, lats = transformer.transform(xs, ys)

        boundary_polygon, _ = await load_and_project_boundary(polygon)
        area_m2 = Polygon(boundary_polygon).area
        area_km2 = area_m2 / 1e6

        farm_nominal_power = len(xs) * 15.0  # Assuming each turbine is 15 MW

        # Build combined properties
        prod_stats = {
            "Weibull_A": data_vars["Weibull_A"]["data"],
            "Weibull_k": data_vars["Weibull_k"]["data"],
            "WS": data_vars["WS"]["data"],
            "WD": data_vars["WD"]["data"],
            "Farm_AEP_Wh": farm_aep * 1e9,
            "Farm_Area_km2": area_km2,
            "WTG_Count": len(xs),
            "Farm_Nominal_Power_W": farm_nominal_power * 1e6,
            "main_direction_prod_W": aep_vs_wd_gwh[int(np.argmax(aep_vs_wd_gwh))] * 1e9 if aep_vs_wd_gwh else None,
            **all_data_area_stats  # Spread the stats here
        }

        farm_id = f"farm-{polygon.id if hasattr(polygon, 'id') else 'N/A'}"
        geojson_return = {
            "type": "FeatureCollection",
            "features": [],
            "properties": {
                "LAYER_NAME": polygon.properties.get("LAYER_NAME", "N/A") if polygon.properties else polygon.properties.get("Empreendimento", "N/A"),
                "prod_stats": prod_stats
            },
            "id": farm_id
        }

        wake_total_loss_w = 0.0
        wake_total_no_wake_power_w = 0.0

        for i in range(len(xs)):
            ws_local = _extract_scalar(data_vars.get("WS", {}).get("data", []), i, default=0.0)
            ws_eff_local = _extract_scalar(data_vars.get("WS_eff", {}).get("data", []), i, default=0.0)
            power_raw = _extract_scalar(data_vars.get("Power", {}).get("data", []), i, default=0.0)
            power_w = max(0.0, power_raw if power_raw > 1e6 else power_raw * 1e3)
            no_wake_power_w = _estimate_no_wake_power_w(ws_local)
            wake_loss_w = max(0.0, no_wake_power_w - power_w)
            wake_loss_pct = (wake_loss_w / no_wake_power_w) * 100.0 if no_wake_power_w > 0 else 0.0

            wake_total_loss_w += wake_loss_w
            wake_total_no_wake_power_w += no_wake_power_w

            feature = {
                "type": "Feature",
                "properties": {
                    "wt": i + 1,
                    "Aerogerador": "IEA_Reference_15MW_240",
                    "Pot": 15.0,
                    "WS": ws_local,
                    "WS_eff": ws_eff_local,
                    "TI_eff": _extract_scalar(data_vars.get("TI_eff", {}).get("data", []), i, default=0.0),
                    "Power": power_w,
                    "CT": _extract_scalar(data_vars.get("CT", {}).get("data", []), i, default=0.0),
                    "h": _extract_scalar(data_vars.get("h", {}).get("data", []), i, default=150.0),
                    "wake_loss_w": round(wake_loss_w, 2),
                    "wake_loss_pct": round(wake_loss_pct, 3),
                    "no_wake_power_w": round(no_wake_power_w, 2),
                    "type": "WTG"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lons[i], lats[i]]
                }
            }

            geojson_return["features"].append(feature)

        wake_total_loss_pct = (wake_total_loss_w / wake_total_no_wake_power_w) * 100.0 if wake_total_no_wake_power_w > 0 else 0.0
        prod_stats.update({
            "wake_total_loss_w": round(wake_total_loss_w, 2),
            "wake_total_loss_pct": round(wake_total_loss_pct, 3)
        })

        geojson_return["features"].append({
            "type": "Feature",
            "properties": {
                "type": "Boundary",
                "Farm_Area_km2": area_km2
            },
            "geometry": polygon.geometry
        })

        return geojson_return

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"External service error: {e.response.text}")
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error in generate_geojson: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def get_point_from_service(geojson_name, polygon):
    url = f"{settings.MAIN_SERVICE}geojsons/query/centroid/{geojson_name}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=polygon.model_dump())
            response.raise_for_status()
            
            data = response.json()
            
            # Decrypt response if encrypted
            if isinstance(data, dict) and "encrypted_data" in data:
                try:
                    encrypted_content = data["encrypted_data"]
                    key = binascii.unhexlify(settings.ENCRYPTION_KEY_HEX)
                    decoded_data = base64.b64decode(encrypted_content)
                    
                    iv = decoded_data[:16]
                    ciphertext = decoded_data[16:]
                    
                    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
                    decryptor = cipher.decryptor()
                    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
                    
                    unpadder = padding.PKCS7(128).unpadder()
                    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
                    
                    return json.loads(plaintext.decode('utf-8'))
                except Exception as e:
                    print(f"Decryption error: {e}")
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error decrypting service response")
            
            return data
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service unavailable: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
        raise exc

