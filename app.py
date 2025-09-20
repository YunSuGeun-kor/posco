import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
import os
import math
import io
from typing import List, Optional, Tuple
import numpy as np
import requests
import json
from datetime import datetime

# =========================
# 기존 설정
# =========================
st.set_page_config(
    page_title="암롤박스 위치 조회 시스템",
    page_icon="📍",
    layout="wide"
)

# -------------------------
# 유틸
# -------------------------
def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlng/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_distance_matrix(points: List[Tuple[float, float]]) -> np.ndarray:
    n = len(points)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = _haversine(points[i][0], points[i][1], points[j][0], points[j][1])
            D[i, j] = D[j, i] = d
    return D

def nearest_neighbor_route(D: np.ndarray, start_idx: int = 0) -> List[int]:
    n = D.shape[0]
    unvisited = set(range(n))
    route = [start_idx]
    unvisited.remove(start_idx)
    cur = start_idx
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        route.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return route

def two_opt(route: List[int], D: np.ndarray, max_iter: int = 2000) -> List[int]:
    best = route[:]
    n = len(best)
    improved = True
    it = 0
    def seg_cost(a,b,c,d): return D[a,b] + D[c,d]
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, n-2):
            for k in range(i+1, n-1):
                a,b = best[i-1], best[i]
                c,d = best[k], best[k+1]
                if seg_cost(a,b,c,d) > seg_cost(a,c,b,d):
                    best[i:k+1] = reversed(best[i:k+1])
                    improved = True
    return best

def route_length(route: List[int], D: np.ndarray) -> float:
    return sum(D[route[i], route[i+1]] for i in range(len(route)-1))

def serialize_route(route_idx: List[int], ids: List[str]) -> List[str]:
    return [ids[i] for i in route_idx]

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

# =========================
# 데이터 로딩
# =========================
@st.cache_data
def load_excel_data(file_path: str, sheet_name: str = "좌표정보") -> Optional[pd.DataFrame]:
    try:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            st.success(f"'{sheet_name}' 시트에서 데이터를 로드했습니다.")
            return df
        except ValueError:
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            available_sheets = excel_file.sheet_names
            st.warning(f"'{sheet_name}' 시트를 찾을 수 없습니다. 사용 가능한 시트: {', '.join(available_sheets)}")
            if available_sheets:
                df = pd.read_excel(file_path, sheet_name=available_sheets[0], engine='openpyxl')
                st.info(f"'{available_sheets[0]}' 시트에서 데이터를 로드했습니다.")
                return df
            else:
                st.error("읽을 수 있는 시트가 없습니다.")
                return None
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {str(e)}")
        if "not a zip file" in str(e).lower():
            st.error("파일이 올바른 Excel 형식이 아닙니다. .xlsx 또는 .xls 파일인지 확인해주세요.")
        elif "no such file" in str(e).lower():
            st.error("파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None

def detect_columns(df: pd.DataFrame) -> dict:
    detected = {'box': None, 'lat': None, 'lng': None}
    for col in df.columns:
        col_clean = str(col).strip()
        if col_clean == '박스번호': detected['box'] = col
        elif col_clean == '위도(DD)': detected['lat'] = col
        elif col_clean == '경도(DD)': detected['lng'] = col
    if not all(detected.values()):
        for col in df.columns:
            low = str(col).lower()
            if detected['box'] is None and any(p in low for p in ['박스번호','box','박스','번호','number','id','no']):
                detected['box'] = col
            if detected['lat'] is None and any(p in low for p in ['위도(dd)','위도','lat','latitude','y']):
                detected['lat'] = col
            if detected['lng'] is None and any(p in low for p in ['경도(dd)','경도','lng','lon','longitude','x']):
                detected['lng'] = col
    return detected

def parse_box_numbers(input_text: str) -> List[str]:
    if not input_text.strip(): return []
    return [num.strip() for num in input_text.split(',') if num.strip()]

def perform_kmeans_clustering(df: pd.DataFrame, lat_col: str, lng_col: str, box_col: str, n_clusters: int) -> tuple[pd.DataFrame, dict]:
    coordinates = df[[lat_col, lng_col]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(coordinates)
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    cluster_info = {}
    for i in range(n_clusters):
        cluster_boxes = df_clustered[df_clustered['cluster'] == i]
        center = kmeans.cluster_centers_[i]
        cluster_info[i] = {
            'center_lat': float(center[0]),
            'center_lng': float(center[1]),
            'box_count': len(cluster_boxes),
            'box_numbers': cluster_boxes[box_col].astype(str).tolist(),
            'boxes_data': cluster_boxes
        }
    return df_clustered, cluster_info

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    return _haversine(lat1, lng1, lat2, lng2)

# =========================
# 지도 생성(기존 함수 약간 수정)
# =========================
def create_map(df: pd.DataFrame, box_col: str, lat_col: str, lng_col: str, 
               filtered_boxes: Optional[List[str]] = None, use_clustering: bool = True, 
               map_style: str = "OpenStreetMap", cluster_count: Optional[int] = None,
               selected_clusters: Optional[list] = None) -> tuple[folium.Map, Optional[dict]]:
    if filtered_boxes:
        df_filtered = df[df[box_col].astype(str).isin(filtered_boxes)]
    else:
        df_filtered = df.copy()
    center_lat, center_lng = 34.926111, 127.764722
    cluster_info = None
    if map_style == "위성지도":
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"; attr = "Esri"
    elif map_style == "지형지도":
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"; attr = "Esri"
    elif map_style == "CartoDB":
        tiles = "CartoDB positron"; attr = None
    else:
        tiles = "OpenStreetMap"; attr = None
    m = folium.Map(location=[center_lat, center_lng], zoom_start=14, tiles=tiles, attr=attr) if attr else folium.Map(location=[center_lat, center_lng], zoom_start=14, tiles=tiles)
    if use_clustering and len(df_filtered) > 5:
        if cluster_count and cluster_count > 0:
            if cluster_count < len(df_filtered):
                df_clustered, cluster_info = perform_kmeans_clustering(df_filtered, lat_col, lng_col, box_col, cluster_count)
                df_filtered = df_clustered
                container = m
            else:
                container = m
        else:
            marker_cluster = MarkerCluster(showCoverageOnHover=True, zoomToBoundsOnClick=True).add_to(m)
            container = marker_cluster
    else:
        container = m
    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7','#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                     '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2','#FAD7A0', '#A9DFBF', '#F9E79F', '#D5A6BD']
    if cluster_info and selected_clusters is not None and len(cluster_info) > 1:
        selected_ids = [int(s.split()[-1]) - 1 for s in selected_clusters]
        if isinstance(df_filtered, pd.DataFrame) and 'cluster' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['cluster'].isin(selected_ids)]
    for _, row in df_filtered.iterrows():
        try:
            lat = float(row[lat_col]); lng = float(row[lng_col]); box_num = str(row[box_col])
            popup_html = f"""
            <div style="font-family: 'Noto Sans KR', sans-serif; font-size: 14px;">
              <h4 style="margin: 5px 0; color: #2E86AB;">📦 박스 {box_num}</h4>
              <p style="margin: 3px 0;"><b>위도:</b> {lat:.6f}</p>
              <p style="margin: 3px 0;"><b>경도:</b> {lng:.6f}</p>
            """
            if 'cluster' in row:
                popup_html += f"<p style='margin: 3px 0;'><b>클러스터:</b> {int(row['cluster']) + 1}</p>"
            add_cols = [c for c in df_filtered.columns if c not in [box_col, lat_col, lng_col, 'cluster']]
            for c in add_cols[:3]:
                v = row.get(c, None)
                if pd.notna(v): popup_html += f"<p style='margin:3px 0;'><b>{c}:</b> {v}</p>"
            popup_html += "</div>"
            if 'cluster' in row:
                color = cluster_colors[int(row['cluster']) % len(cluster_colors)]
            elif '상태' in df_filtered.columns and pd.notna(row['상태']):
                s = str(row['상태']).lower()
                if '정상' in s: color = '#2ECC71'
                elif '점검' in s: color = '#F39C12'
                elif '수리' in s: color = '#E74C3C'
                else: color = '#3498DB'
            else:
                color = '#FF00FF'
            folium.Circle(
                location=[lat,lng], radius=15, popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"📦 박스 {box_num}", color=color, fill=True, fill_color=color,
                weight=11.5, fill_opacity=0.7, opacity=0.9
            ).add_to(container)
        except Exception as e:
            st.warning(f"마커 생성 실패: {e}")
            continue
    if cluster_info:
        legend_html = '''
        <div style="position: fixed; top: 50%; left: 10px; width: 180px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2); transform: translateY(-50%); font-family: 'Noto Sans KR', Arial, sans-serif;">
        <h4 style="margin: 0 0 10px 0; text-align: center; color: #333; font-size: 16px;">클러스터 범례</h4>
        '''
        for cid in sorted(cluster_info.keys()):
            color = cluster_colors[cid % len(cluster_colors)]
            box_count = cluster_info[cid]['box_count']
            legend_html += f'''
            <div style="margin: 8px 0; padding: 5px; background-color: #f9f9f9; border-radius: 3px;">
              <div style="display: flex; align-items: center; margin-bottom: 2px;">
                <div style="width: 18px; height: 18px; background-color: {color}; border: 2px solid #333; border-radius: 50%; margin-right: 8px;"></div>
                <span style="color: #333; font-weight: bold; font-size: 13px;">클러스터 {cid + 1}</span>
              </div>
              <div style="margin-left: 26px;"><span style="color: #666; font-size: 11px;">박스 {box_count}개</span></div>
            </div>'''
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
    return m, cluster_info

# =========================
# AI 보조 경로 재조정
# =========================
def ai_refine_route(
    api_key: str,
    candidate_order: List[str],
    df_sub: pd.DataFrame,
    box_col: str,
    extra_cols: List[str],
    max_distance_km: float,
    start_label: str
) -> Optional[List[str]]:
    """
    OpenAI에 메타데이터(상태/우선순위/적재량 등)를 전달해 JSON 경로를 제안받음.
    반환: 박스번호 문자열 리스트. 실패 시 None.
    """
    # 샘플 메타데이터 축약
    preview_rows = []
    for _, r in df_sub.iterrows():
        item = {"box": str(r[box_col])}
        for c in extra_cols:
            val = r.get(c, None)
            if pd.notna(val): item[c] = val if isinstance(val, (int,float)) else str(val)[:60]
        preview_rows.append(item)

    system = (
        "You are a logistics optimizer. You receive a candidate pickup order and metadata "
        "for waste boxes. You must output ONLY a JSON list of box IDs (strings) in the "
        "optimized visiting order that respects the starting point implicitly given by the first element. "
        "Constraints: prefer higher priority boxes first (우선순위, 상태=점검/수리 시 가중치↑), "
        "try not to increase route length by more than 20% of provided baseline distance. "
        "No commentary. JSON only."
    )
    user = {
        "start_label": start_label,
        "baseline_km": round(max_distance_km, 3),
        "candidate_order": candidate_order,
        "metadata_columns": extra_cols,
        "items": preview_rows
    }

    try:
        # 표준 Chat Completions REST 호출 (requests 사용)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            st.warning(f"OpenAI API 오류: {resp.status_code} {resp.text[:200]}")
            return None
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        # 데이터 형태 유연 처리: {"order": [...]} 또는 직접 리스트
        if isinstance(data, dict) and "order" in data and isinstance(data["order"], list):
            out = [str(x) for x in data["order"]]
        elif isinstance(data, list):
            out = [str(x) for x in data]
        else:
            st.warning("AI 응답 형식이 올바르지 않습니다.")
            return None
        # 후보에 없는 박스 제거 및 순서 보정
        cand_set = set(candidate_order)
        out = [x for x in out if x in cand_set]
        # 누락된 박스는 후보 순서대로 뒤에 추가
        missing = [x for x in candidate_order if x not in out]
        return out + missing
    except Exception as e:
        st.warning(f"AI 경로 재조정 실패: {e}")
        return None

# =========================
# 메인
# =========================
def main():
    st.title("📍 암롤박스 위치 조회 시스템")
    st.markdown("---")

    # ---------- 사이드바: OpenAI Key ----------
    with st.sidebar:
        st.markdown("### 🔐 OpenAI 설정")
        api_key_input = st.text_input("OpenAI API Key", type="password", help="환경변수 OPENAI_API_KEY 사용 권장")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input

    excel_file = "box_locations.xlsx"
    github_raw_url = "https://raw.githubusercontent.com/YunSuGeun-kor/posco/main/box_locations.xlsx"

    if not os.path.exists(excel_file):
        try:
            with st.spinner("Github에서 엑셀 파일을 불러오는 중입니다..."):
                response = requests.get(github_raw_url, timeout=15)
                if response.status_code == 200:
                    excel_bytes = io.BytesIO(response.content)
                    df = load_excel_data(excel_bytes, "좌표정보")
                else:
                    raise Exception("GitHub에서 파일을 불러오지 못했습니다.")
        except Exception:
            st.error(f"엑셀 파일 '{excel_file}'을 찾을 수 없고, GitHub 저장소에서도 불러오지 못했습니다. 파일을 업로드해주세요.")
            uploaded_file = st.file_uploader("엑셀 파일 선택", type=['xlsx','xls'], help="암롤박스 위치 정보가 포함된 엑셀 파일을 업로드하세요.")
            if uploaded_file is not None:
                with open(excel_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("파일이 업로드되었습니다!")
                st.rerun()
            else:
                st.stop()
    else:
        with st.spinner("데이터를 로딩중입니다..."):
            df = load_excel_data(excel_file, "좌표정보")
        if df is None:
            st.stop()

    st.success(f"✅ 데이터 로딩 완료: {len(df)}개의 레코드")

    # 컬럼 감지 및 선택
    detected = detect_columns(df)
    st.subheader("🔧 컬럼 설정")
    c1,c2,c3 = st.columns(3)
    with c1:
        box_col = st.selectbox("박스 번호 컬럼", options=df.columns.tolist(),
                               index=df.columns.tolist().index(detected['box']) if detected['box'] else 0)
    with c2:
        lat_col = st.selectbox("위도 컬럼", options=df.columns.tolist(),
                               index=df.columns.tolist().index(detected['lat']) if detected['lat'] else 0)
    with c3:
        lng_col = st.selectbox("경도 컬럼", options=df.columns.tolist(),
                               index=df.columns.tolist().index(detected['lng']) if detected['lng'] else 0)

    if not all([box_col, lat_col, lng_col]):
        st.error("모든 컬럼을 선택해주세요.")
        st.stop()

    df_clean = df.dropna(subset=[lat_col, lng_col]).copy()
    # 좌표 안전 변환
    df_clean[lat_col] = df_clean[lat_col].apply(safe_float)
    df_clean[lng_col] = df_clean[lng_col].apply(safe_float)
    df_clean = df_clean.dropna(subset=[lat_col, lng_col])

    if len(df_clean) == 0:
        st.error("유효한 좌표 데이터가 없습니다.")
        st.stop()

    st.info(f"유효한 좌표를 가진 박스: {len(df_clean)}개")

    # 필터
    st.subheader("🔍 박스 필터링")
    fc1, fc2 = st.columns([3,1])
    with fc1:
        box_input = st.text_input("조회할 박스 번호 입력 (쉼표로 구분)", placeholder="예: 101, 205, 333")
    with fc2:
        st.button("🔍 조회", type="primary")
    filtered_boxes = parse_box_numbers(box_input) if box_input else None

    if filtered_boxes:
        st.info(f"필터링된 박스: {', '.join(filtered_boxes)} ({len(filtered_boxes)}개)")
        exist = df_clean[box_col].astype(str).tolist()
        missing = [b for b in filtered_boxes if b not in exist]
        if missing:
            st.warning(f"다음 박스는 데이터에 없습니다: {', '.join(missing)}")
    else:
        st.info(f"전체 박스 표시: {len(df_clean)}개")

    # 지도 옵션
    st.subheader("⚙️ 지도 옵션")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        map_type = st.selectbox("지도 종류", options=["일반지도","위성지도","지형지도","CartoDB"], index=0)
    with mc2:
        use_clustering = st.checkbox("마커 클러스터링 사용", value=True)
    with mc3:
        cluster_mode = st.radio("클러스터링 모드", options=["자동","수동"], index=0, disabled=not use_clustering)
        cluster_count = st.number_input("클러스터 개수", min_value=1, max_value=50, value=3) if (use_clustering and cluster_mode=="수동") else None
    with mc4:
        show_distances = st.checkbox("박스 간 거리 계산", value=False)

    # 임시 지도(클러스터 UI용)
    _, cluster_info = create_map(df_clean, box_col, lat_col, lng_col, filtered_boxes, use_clustering, map_type, cluster_count)

    selected_clusters = None
    if use_clustering and cluster_info and len(cluster_info) > 1:
        labels = [f"클러스터 {i+1}" for i in range(len(cluster_info))]
        if 'selected_clusters' not in st.session_state:
            st.session_state['selected_clusters'] = labels.copy()
        b1,b2 = st.columns(2)
        with b1:
            if st.button("전체 선택"): st.session_state['selected_clusters'] = labels.copy()
        with b2:
            if st.button("전체 취소"): st.session_state['selected_clusters'] = []
        selected_clusters = st.multiselect("표시할 클러스터 선택", options=labels, default=st.session_state['selected_clusters'], key="selected_clusters")

    folium_map, cluster_info = create_map(df_clean, box_col, lat_col, lng_col, filtered_boxes, use_clustering, map_type, cluster_count, selected_clusters)

    st.subheader("🗺️ 위치 지도")
    st.markdown("""
    <style>
      .custom-map-wrapper { padding-right: 200px; }
      iframe { border-radius: 8px; pointer-events: auto !important; }
      @media (max-width: 768px) { .custom-map-wrapper { padding-right: 150px; } iframe { max-width: 70vw !important; } }
    </style>
    <div class="custom-map-wrapper">
    """, unsafe_allow_html=True)

    try:
        map_data = st_folium(folium_map, width=None, height=700, returned_objects=["last_object_clicked"])
        if map_data["last_object_clicked"]:
            clicked = map_data["last_object_clicked"]
            st.success(f"선택된 위치: 위도 {clicked['lat']:.6f}, 경도 {clicked['lng']:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"지도 생성 중 오류: {e}")

    # =========================
    # 수거 경로 최적화
    # =========================
    st.subheader("🛣️ 수거 경로 최적화")
    target_df = df_clean[df_clean[box_col].astype(str).isin(filtered_boxes)] if filtered_boxes else df_clean.copy()

    # 출발지 옵션
    start_mode = st.radio("출발지 선택", ["소각로(기본)","임의 좌표 입력","특정 박스에서 시작"], horizontal=True)
    default_origin = (34.926111, 127.764722)  # 광양제철소 소각로
    if start_mode == "임의 좌표 입력":
        oc1, oc2 = st.columns(2)
        with oc1: start_lat = st.number_input("출발 위도", value=default_origin[0], format="%.6f")
        with oc2: start_lng = st.number_input("출발 경도", value=default_origin[1], format="%.6f")
        origin = (start_lat, start_lng)
        start_label = "custom_origin"
    elif start_mode == "특정 박스에서 시작":
        start_box = st.selectbox("시작 박스 선택", options=target_df[box_col].astype(str).tolist())
        row = target_df[target_df[box_col].astype(str)==start_box].iloc[0]
        origin = (float(row[lat_col]), float(row[lng_col]))
        start_label = f"box:{start_box}"
    else:
        origin = default_origin
        start_label = "incinerator"

    # AI 보조 의사결정
    ai_help = st.checkbox("AI 보조 의사결정 사용(상태/우선순위 반영)", value=False,
                         help="열 이름 예시: '우선순위', '상태', '적재량' 등")

    # 후보 메타 컬럼 선택
    meta_cols = []
    if ai_help:
        candidates = [c for c in target_df.columns if c not in [box_col, lat_col, lng_col, 'cluster']]
        meta_cols = st.multiselect("의사결정에 반영할 컬럼", options=candidates, default=[c for c in candidates if c in ["우선순위","상태","적재량"]])

    calc_btn = st.button("🚚 경로 계산")

    route_result = None
    if calc_btn:
        if len(target_df) < 1:
            st.warning("대상 박스가 없습니다.")
        else:
            # 순서: origin → 박스들
            ids = target_df[box_col].astype(str).tolist()
            coords = [(float(r[lat_col]), float(r[lng_col])) for _, r in target_df.iterrows()]

            # origin을 0번으로 붙여서 거리행렬 생성
            points = [origin] + coords
            D = build_distance_matrix(points)

            # 최근접 시작은 0 이후 노드에서 시작 → 구현을 위해 0을 고정, 나머지 경로 최적화
            # 0은 origin. route는 0->... 형식으로 출력
            # 내부 인덱스: 0=origin, 1..n=박스
            # 최근접은 0에서 시작하여 나머지 방문
            nn = nearest_neighbor_route(D, start_idx=0)
            # 2-opt로 개선
            opt_idx = two_opt(nn, D)
            # origin 제외한 순서로 환산
            visit_idx = [i for i in opt_idx if i != 0]
            base_km = route_length(opt_idx, D)
            ordered_boxes = serialize_route(visit_idx, ids)

            st.success(f"기본 경로 총거리: {base_km:.2f} km")

            # AI 보조
            final_order = ordered_boxes[:]
            if ai_help:
                key = os.environ.get("OPENAI_API_KEY") or ""
                if not key:
                    st.warning("OpenAI API Key가 필요합니다. 사이드바에서 입력하세요.")
                else:
                    df_sub = target_df.set_index(target_df[box_col].astype(str)).loc[ordered_boxes].reset_index(drop=True)
                    ai_order = ai_refine_route(
                        api_key=key,
                        candidate_order=ordered_boxes,
                        df_sub=df_sub,
                        box_col=box_col,
                        extra_cols=meta_cols,
                        max_distance_km=base_km,
                        start_label=start_label
                    )
                    if ai_order:
                        # AI 순서를 거리 기준으로 검증
                        # ai_order를 내부 인덱스로 변환
                        idx_map = {bid:i+1 for i,bid in enumerate(ids)}  # +1: origin 보정
                        try:
                            ai_visit_idx = [idx_map[b] for b in ai_order]
                            ai_route = [0] + ai_visit_idx
                            ai_km = route_length(ai_route, D)
                            st.info(f"AI 재조정 경로 총거리: {ai_km:.2f} km")
                            # 20% 이내 증가 허용
                            if ai_km <= base_km * 1.2 + 1e-6:
                                final_order = ai_order
                                st.success("AI 경로 적용됨(거리 증가 20% 이내).")
                            else:
                                st.warning("AI 경로가 거리 제한을 초과하여 기본 경로를 유지합니다.")
                        except Exception as e:
                            st.warning(f"AI 경로 검증 실패: {e}")

            # 최종 경로 지도 표시
            # 순번 라벨링을 위해 dict 구성
            order_pos = []
            id_to_coord = {str(r[box_col]): (float(r[lat_col]), float(r[lng_col])) for _, r in target_df.iterrows()}
            for b in final_order:
                order_pos.append((b, id_to_coord[b]))

            # 지도 생성
            route_map = folium.Map(location=[origin[0], origin[1]], zoom_start=14, tiles="OpenStreetMap")
            # 출발지 마커
            folium.Marker(location=[origin[0], origin[1]],
                          icon=folium.Icon(color="green", icon="play"),
                          tooltip="출발지").add_to(route_map)

            # 경로 마커 및 라벨
            poly = [origin]
            for i, (bid, (la, lo)) in enumerate(order_pos, start=1):
                folium.Marker(
                    location=[la, lo],
                    tooltip=f"{i}. 박스 {bid}",
                    icon=folium.DivIcon(html=f"""<div style="font-size:12px;color:#000;background:#fff;border:1px solid #333;border-radius:10px;padding:2px 6px;">{i}</div>""")
                ).add_to(route_map)
                poly.append((la, lo))

            # 폴리라인
            folium.PolyLine(locations=poly, weight=5, opacity=0.8).add_to(route_map)

            st.markdown("**최종 경로 지도**")
            st_folium(route_map, width=None, height=600)

            # 결과 표
            dist_rows = []
            total_km = 0.0
            prev = origin
            for i, (bid, (la, lo)) in enumerate(order_pos, start=1):
                seg = _haversine(prev[0], prev[1], la, lo)
                total_km += seg
                dist_rows.append({"순번": i, "박스번호": bid, "세그먼트거리(km)": round(seg,2), "누적거리(km)": round(total_km,2)})
                prev = (la, lo)
            res_df = pd.DataFrame(dist_rows)
            st.dataframe(res_df, use_container_width=True)
            st.metric("총 주행거리", f"{total_km:.2f} km")

            # 내보내기
            colx, coly = st.columns(2)
            with colx:
                csv = res_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("경로 CSV 다운로드", data=csv,
                                   file_name=f"수거경로_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")
            with coly:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    res_df.to_excel(w, index=False, sheet_name="수거경로")
                st.download_button("경로 엑셀 다운로드", data=buf.getvalue(),
                                   file_name=f"수거경로_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # 거리표 기능 유지
    if show_distances and filtered_boxes and len(filtered_boxes) > 1:
        st.subheader("📏 박스 간 거리")
        distance_df = df_clean[df_clean[box_col].astype(str).isin(filtered_boxes)]
        if len(distance_df) > 1:
            distances = []
            indices = list(distance_df.index)
            for i_idx, i in enumerate(indices):
                for j_idx, j in enumerate(indices):
                    if i_idx < j_idx:
                        r1 = distance_df.loc[i]; r2 = distance_df.loc[j]
                        b1, b2 = str(r1[box_col]), str(r2[box_col])
                        lat1, lng1 = float(r1[lat_col]), float(r1[lng_col])
                        lat2, lng2 = float(r2[lat_col]), float(r2[lng_col])
                        d = calculate_distance(lat1,lng1,lat2,lng2)
                        distances.append({'박스1': b1, '박스2': b2, '거리(km)': round(d, 2)})
            if distances:
                ddf = pd.DataFrame(distances).sort_values('거리(km)')
                st.dataframe(ddf, use_container_width=True)
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("평균 거리", f"{ddf['거리(km)'].mean():.2f} km")
                with col2: st.metric("최단 거리", f"{ddf['거리(km)'].min():.2f} km")
                with col3: st.metric("최장 거리", f"{ddf['거리(km)'].max():.2f} km")

    # 내보내기(원본/필터 데이터)
    st.subheader("💾 데이터 내보내기")
    export_df = df_clean[df_clean[box_col].astype(str).isin(filtered_boxes)] if filtered_boxes else df_clean
    export_title = f"{'필터링된' if filtered_boxes else '전체'} 박스 데이터 ({len(export_df)}개)"
    cex1, cex2 = st.columns(2)
    with cex1:
        if st.button("📊 Excel 파일로 내보내기"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='암롤박스_데이터', index=False)
            st.download_button("💾 Excel 파일 다운로드", data=buffer.getvalue(),
                               file_name=f"암롤박스_위치정보_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with cex2:
        if st.button("📄 CSV 파일로 내보내기"):
            csv2 = export_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("💾 CSV 파일 다운로드", data=csv2,
                               file_name=f"암롤박스_위치정보_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")

    with st.expander("📊 데이터 미리보기"):
        st.write(export_title)
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("표시된 박스 수", len(export_df))
        with c2:
            if '상태' in export_df.columns:
                normal_count = len(export_df[export_df['상태'] == '정상'])
                st.metric("정상 상태 박스", normal_count)
        with c3:
            if len(export_df) > 0:
                lat_range = export_df[lat_col].max() - export_df[lat_col].min()
                st.metric("위도 범위", f"{lat_range:.4f}°")
        if cluster_info:
            st.subheader("🎯 클러스터 요약")
            summary = []
            for i, info in cluster_info.items():
                summary.append({'클러스터': i+1, '박스 개수': info['box_count'],
                                '중심 위도': round(info['center_lat'],6),
                                '중심 경도': round(info['center_lng'],6),
                                '포함 박스': ', '.join(info['box_numbers'][:5]) + ('...' if len(info['box_numbers'])>5 else '')})
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
        st.subheader("📋 전체 데이터")
        st.dataframe(export_df, use_container_width=True)

if __name__ == "__main__":
    main()
