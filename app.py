import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
import os, math, io, json, requests
from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime

st.set_page_config(page_title="암롤박스 위치 조회 시스템", page_icon="📍", layout="wide")

# ---------- 유틸 ----------
def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1); dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlng/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_distance_matrix(points: List[Tuple[float, float]]) -> np.ndarray:
    n = len(points); D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = _haversine(points[i][0], points[i][1], points[j][0], points[j][1])
            D[i, j] = D[j, i] = d
    return D

def nearest_neighbor_route(D: np.ndarray, start_idx: int = 0) -> List[int]:
    n = D.shape[0]
    unvisited = set(range(n)); route = [start_idx]; unvisited.remove(start_idx); cur = start_idx
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        route.append(nxt); unvisited.remove(nxt); cur = nxt
    return route

def two_opt(route: List[int], D: np.ndarray, max_iter: int = 2000) -> List[int]:
    best = route[:]; n = len(best); it = 0; improved = True
    def seg_cost(a,b,c,d): return D[a,b] + D[c,d]
    while improved and it < max_iter:
        improved = False; it += 1
        for i in range(1, n-2):
            for k in range(i+1, n-1):
                a,b = best[i-1], best[i]; c,d = best[k], best[k+1]
                if seg_cost(a,b,c,d) > seg_cost(a,c,b,d):
                    best[i:k+1] = reversed(best[i:k+1]); improved = True
    return best

def route_length(route: List[int], D: np.ndarray) -> float:
    return sum(D[route[i], route[i+1]] for i in range(len(route)-1))

def serialize_route(route_idx: List[int], ids: List[str]) -> List[str]:
    # route_idx는 points 인덱스(0=origin, 1..n=박스). ids는 0..n-1
    return [ids[i-1] for i in route_idx]  # <-- 핵심 보정

def safe_float(x) -> Optional[float]:
    try: return float(x)
    except Exception: return None

# ---------- 데이터 ----------
@st.cache_data
def load_excel_data(file_path: str, sheet_name: str = "좌표정보") -> Optional[pd.DataFrame]:
    try:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            st.success(f"'{sheet_name}' 시트에서 데이터를 로드했습니다."); return df
        except ValueError:
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            sheets = excel_file.sheet_names
            st.warning(f"'{sheet_name}' 시트를 찾을 수 없습니다. 사용 가능한 시트: {', '.join(sheets)}")
            if sheets:
                df = pd.read_excel(file_path, sheet_name=sheets[0], engine='openpyxl')
                st.info(f"'{sheets[0]}' 시트에서 데이터를 로드했습니다."); return df
            st.error("읽을 수 있는 시트가 없습니다."); return None
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류: {str(e)}")
        if "not a zip file" in str(e).lower():
            st.error(".xlsx 또는 .xls 파일인지 확인해주세요.")
        elif "no such file" in str(e).lower():
            st.error("파일을 찾을 수 없습니다.")
        return None

def detect_columns(df: pd.DataFrame) -> dict:
    detected = {'box': None, 'lat': None, 'lng': None}
    for col in df.columns:
        c = str(col).strip()
        if c == '박스번호': detected['box']=col
        elif c == '위도(DD)': detected['lat']=col
        elif c == '경도(DD)': detected['lng']=col
    if not all(detected.values()):
        for col in df.columns:
            low = str(col).lower()
            if detected['box'] is None and any(p in low for p in ['박스번호','box','박스','번호','number','id','no']): detected['box']=col
            if detected['lat'] is None and any(p in low for p in ['위도(dd)','위도','lat','latitude','y']): detected['lat']=col
            if detected['lng'] is None and any(p in low for p in ['경도(dd)','경도','lng','lon','longitude','x']): detected['lng']=col
    return detected

def parse_box_numbers(text: str) -> List[str]:
    if not text or not text.strip(): return []
    return [t.strip() for t in text.split(',') if t.strip()]

def perform_kmeans_clustering(df: pd.DataFrame, lat_col: str, lng_col: str, box_col: str, n_clusters: int) -> tuple[pd.DataFrame, dict]:
    coordinates = df[[lat_col, lng_col]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(coordinates)
    dfc = df.copy(); dfc['cluster'] = labels
    info = {}
    for i in range(n_clusters):
        sub = dfc[dfc['cluster']==i]; center = kmeans.cluster_centers_[i]
        info[i] = {'center_lat': float(center[0]), 'center_lng': float(center[1]), 'box_count': len(sub),
                   'box_numbers': sub[box_col].astype(str).tolist(), 'boxes_data': sub}
    return dfc, info

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    return _haversine(lat1, lng1, lat2, lng2)

# ---------- 지도 ----------
def create_map(df: pd.DataFrame, box_col: str, lat_col: str, lng_col: str, 
               filtered_boxes: Optional[List[str]] = None, use_clustering: bool = True, 
               map_style: str = "OpenStreetMap", cluster_count: Optional[int] = None,
               selected_clusters: Optional[list] = None) -> tuple[folium.Map, Optional[dict]]:
    df_filtered = df[df[box_col].astype(str).isin(filtered_boxes)] if filtered_boxes else df.copy()
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
        if cluster_count and cluster_count > 0 and cluster_count < len(df_filtered):
            df_filtered, cluster_info = perform_kmeans_clustering(df_filtered, lat_col, lng_col, box_col, cluster_count)
            container = m
        else:
            container = MarkerCluster(showCoverageOnHover=True, zoomToBoundsOnClick=True).add_to(m)
    else:
        container = m
    cluster_colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD','#98D8C8','#F7DC6F','#BB8FCE','#85C1E9','#F8C471','#82E0AA','#F1948A','#85C1E9','#D7BDE2','#FAD7A0','#A9DFBF','#F9E79F','#D5A6BD']
    if cluster_info and selected_clusters is not None and len(cluster_info) > 1:
        selected_ids = [int(s.split()[-1]) - 1 for s in selected_clusters]
        if 'cluster' in df_filtered.columns: df_filtered = df_filtered[df_filtered['cluster'].isin(selected_ids)]
    for _, row in df_filtered.iterrows():
        try:
            lat = float(row[lat_col]); lng = float(row[lng_col]); box_num = str(row[box_col])
            popup_html = f"<div style='font-family:Noto Sans KR;font-size:14px;'><h4 style='margin:5px 0;color:#2E86AB'>📦 박스 {box_num}</h4><p><b>위도:</b> {lat:.6f}</p><p><b>경도:</b> {lng:.6f}</p>"
            if 'cluster' in row: popup_html += f"<p><b>클러스터:</b> {int(row['cluster'])+1}</p>"
            add_cols = [c for c in df_filtered.columns if c not in [box_col, lat_col, lng_col, 'cluster']]
            for c in add_cols[:3]:
                v = row.get(c, None)
                if pd.notna(v): popup_html += f"<p><b>{c}:</b> {v}</p>"
            popup_html += "</div>"
            if 'cluster' in row:
                color = cluster_colors[int(row['cluster']) % len(cluster_colors)]
            elif '상태' in df_filtered.columns and pd.notna(row['상태']):
                s = str(row['상태']).lower()
                color = '#2ECC71' if '정상' in s else '#F39C12' if '점검' in s else '#E74C3C' if '수리' in s else '#3498DB'
            else:
                color = '#FF00FF'
            folium.Circle(location=[lat,lng], radius=15, popup=folium.Popup(popup_html, max_width=300),
                          tooltip=f"📦 박스 {box_num}", color=color, fill=True, fill_color=color,
                          weight=11.5, fill_opacity=0.7, opacity=0.9).add_to(container)
        except Exception as e:
            st.warning(f"마커 생성 실패: {e}")
    if cluster_info:
        html = '<div style="position:fixed;top:50%;left:10px;width:180px;background:#fff;border:2px solid grey;z-index:9999;font-size:14px;padding:10px;border-radius:5px;box-shadow:0 0 15px rgba(0,0,0,0.2);transform:translateY(-50%);font-family:Noto Sans KR,Arial;"><h4 style="margin:0 0 10px;text-align:center;color:#333;font-size:16px;">클러스터 범례</h4>'
        for cid in sorted(cluster_info.keys()):
            color = cluster_colors[cid % len(cluster_colors)]; cnt = cluster_info[cid]['box_count']
            html += f"<div style='margin:8px 0;padding:5px;background:#f9f9f9;border-radius:3px;'><div style='display:flex;align-items:center;margin-bottom:2px;'><div style='width:18px;height:18px;background:{color};border:2px solid #333;border-radius:50%;margin-right:8px;'></div><span style='color:#333;font-weight:bold;font-size:13px;'>클러스터 {cid+1}</span></div><div style='margin-left:26px;'><span style='color:#666;font-size:11px;'>박스 {cnt}개</span></div></div>"
        html += '</div>'; m.get_root().html.add_child(folium.Element(html))
    return m, cluster_info

# ---------- AI 경로 재조정 ----------
def ai_refine_route(api_key: str, candidate_order: List[str], df_sub: pd.DataFrame,
                    box_col: str, extra_cols: List[str], max_distance_km: float, start_label: str) -> Optional[List[str]]:
    preview_rows = []
    for _, r in df_sub.iterrows():
        item = {"box": str(r[box_col])}
        for c in extra_cols:
            val = r.get(c, None)
            if pd.notna(val): item[c] = val if isinstance(val,(int,float)) else str(val)[:60]
        preview_rows.append(item)
    system = ("You are a logistics optimizer. Output ONLY a JSON list under key 'order' with box IDs in optimized visiting order. "
              "Prefer higher priority, 상태=점검/수리 first, do not increase distance >20% vs baseline. No commentary.")
    user = {"start_label": start_label, "baseline_km": round(max_distance_km,3),
            "candidate_order": candidate_order, "metadata_columns": extra_cols, "items": preview_rows}
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": "gpt-4o-mini",
                   "messages": [{"role":"system","content":system},
                                {"role":"user","content":json.dumps(user, ensure_ascii=False)}],
                   "temperature": 0.1,
                   "response_format": {"type":"json_object"}}
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            st.warning(f"OpenAI API 오류: {resp.status_code} {resp.text[:200]}"); return None
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        out = data["order"] if isinstance(data, dict) and isinstance(data.get("order"), list) else (data if isinstance(data, list) else None)
        if out is None: st.warning("AI 응답 형식 오류"); return None
        out = [str(x) for x in out]
        cand = set(candidate_order)
        out = [x for x in out if x in cand] + [x for x in candidate_order if x not in out]
        return out
    except Exception as e:
        st.warning(f"AI 경로 재조정 실패: {e}"); return None

# ---------- 메인 ----------
def main():
    st.title("📍 암롤박스 위치 조회 시스템"); st.markdown("---")

    # OpenAI 키: secrets 사용
    # Streamlit Cloud/Community: Settings → Secrets → {"OPENAI_API_KEY":"sk-..."}
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

    excel_file = "box_locations.xlsx"
    github_raw_url = "https://raw.githubusercontent.com/YunSuGeun-kor/posco/main/box_locations.xlsx"

    if not os.path.exists(excel_file):
        try:
            with st.spinner("Github에서 엑셀 파일을 불러오는 중입니다..."):
                r = requests.get(github_raw_url, timeout=15)
                if r.status_code == 200:
                    excel_bytes = io.BytesIO(r.content); df = load_excel_data(excel_bytes, "좌표정보")
                else:
                    raise RuntimeError("GitHub에서 파일을 불러오지 못했습니다.")
        except Exception:
            st.error(f"엑셀 '{excel_file}' 없음. 파일을 업로드해주세요.")
            up = st.file_uploader("엑셀 파일 선택", type=['xlsx','xls'])
            if up is not None:
                with open(excel_file, "wb") as f: f.write(up.getbuffer())
                st.success("업로드 완료"); st.rerun()
            else: st.stop()
    else:
        with st.spinner("데이터를 로딩중입니다..."):
            df = load_excel_data(excel_file, "좌표정보")
        if df is None: st.stop()

    st.success(f"✅ 데이터 로딩 완료: {len(df)}개의 레코드")

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

    if not all([box_col, lat_col, lng_col]): st.error("모든 컬럼을 선택해주세요."); st.stop()

    df_clean = df.dropna(subset=[lat_col, lng_col]).copy()
    df_clean[lat_col] = df_clean[lat_col].apply(safe_float); df_clean[lng_col] = df_clean[lng_col].apply(safe_float)
    df_clean = df_clean.dropna(subset=[lat_col, lng_col])
    if len(df_clean) == 0: st.error("유효한 좌표 데이터가 없습니다."); st.stop()
    st.info(f"유효한 좌표를 가진 박스: {len(df_clean)}개")

    st.subheader("🔍 박스 필터링")
    fc1, fc2 = st.columns([3,1])
    with fc1: box_input = st.text_input("조회할 박스 번호 입력 (쉼표로 구분)", placeholder="예: 101, 205, 333")
    with fc2: st.button("🔍 조회", type="primary")
    filtered_boxes = parse_box_numbers(box_input) if box_input else None

    if filtered_boxes:
        st.info(f"필터링된 박스: {', '.join(filtered_boxes)} ({len(filtered_boxes)}개)")
        exist = df_clean[box_col].astype(str).tolist()
        missing = [b for b in filtered_boxes if b not in exist]
        if missing: st.warning(f"데이터 없음: {', '.join(missing)}")
    else:
        st.info(f"전체 박스 표시: {len(df_clean)}개")

    st.subheader("⚙️ 지도 옵션")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: map_type = st.selectbox("지도 종류", options=["일반지도","위성지도","지형지도","CartoDB"], index=0)
    with mc2: use_clustering = st.checkbox("마커 클러스터링 사용", value=True)
    with mc3:
        cluster_mode = st.radio("클러스터링 모드", options=["자동","수동"], index=0, disabled=not use_clustering)
        cluster_count = st.number_input("클러스터 개수", min_value=1, max_value=50, value=3) if (use_clustering and cluster_mode=="수동") else None
    with mc4: show_distances = st.checkbox("박스 간 거리 계산", value=False)

    _, cluster_info = create_map(df_clean, box_col, lat_col, lng_col, filtered_boxes, use_clustering, map_type, cluster_count)

    selected_clusters = None
    if use_clustering and cluster_info and len(cluster_info) > 1:
        labels = [f"클러스터 {i+1}" for i in range(len(cluster_info))]
        if 'selected_clusters' not in st.session_state: st.session_state['selected_clusters'] = labels.copy()
        b1,b2 = st.columns(2)
        with b1:
            if st.button("전체 선택"): st.session_state['selected_clusters'] = labels.copy()
        with b2:
            if st.button("전체 취소"): st.session_state['selected_clusters'] = []
        selected_clusters = st.multiselect("표시할 클러스터 선택", options=labels, default=st.session_state['selected_clusters'], key="selected_clusters")

    folium_map, cluster_info = create_map(df_clean, box_col, lat_col, lng_col, filtered_boxes, use_clustering, map_type, cluster_count, selected_clusters)

    st.subheader("🗺️ 위치 지도")
    st.markdown("""
    <style>.custom-map-wrapper{padding-right:200px;} iframe{border-radius:8px;pointer-events:auto!important;}</style>
    <div class="custom-map-wrapper">""", unsafe_allow_html=True)
    try:
        map_data = st_folium(folium_map, width=None, height=700, returned_objects=["last_object_clicked"])
        clicked = (map_data or {}).get("last_object_clicked")
        if clicked: st.success(f"선택된 위치: 위도 {clicked.get('lat',0):.6f}, 경도 {clicked.get('lng',0):.6f}")
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"지도 생성 중 오류: {e}")

    # ---------- 수거 경로 최적화 ----------
    st.subheader("🛣️ 수거 경로 최적화")
    target_df = df_clean[df_clean[box_col].astype(str).isin(filtered_boxes)] if filtered_boxes else df_clean.copy()
    if len(target_df) == 0:
        st.info("대상 박스가 없습니다."); return

    start_mode = st.radio("출발지 선택", ["소각로(기본)","임의 좌표 입력","특정 박스에서 시작"], horizontal=True)
    default_origin = (34.926111, 127.764722)
    if start_mode == "임의 좌표 입력":
        oc1, oc2 = st.columns(2)
        with oc1: start_lat = st.number_input("출발 위도", value=default_origin[0], format="%.6f")
        with oc2: start_lng = st.number_input("출발 경도", value=default_origin[1], format="%.6f")
        origin = (start_lat, start_lng); start_label = "custom_origin"
    elif start_mode == "특정 박스에서 시작":
        start_box = st.selectbox("시작 박스 선택", options=target_df[box_col].astype(str).tolist())
        row = target_df[target_df[box_col].astype(str)==start_box].iloc[0]
        origin = (float(row[lat_col]), float(row[lng_col])); start_label = f"box:{start_box}"
    else:
        origin = default_origin; start_label = "incinerator"

    ai_help = st.checkbox("AI 보조 의사결정 사용(상태/우선순위 반영)", value=False,
                          help="Secrets에 OPENAI_API_KEY 설정 필요")

    meta_cols = []
    if ai_help:
        candidates = [c for c in target_df.columns if c not in [box_col, lat_col, lng_col, 'cluster']]
        meta_cols = st.multiselect("의사결정에 반영할 컬럼", options=candidates,
                                   default=[c for c in candidates if c in ["우선순위","상태","적재량"]])

    if st.button("🚚 경로 계산"):
        ids = target_df[box_col].astype(str).tolist()
        coords = [(float(r[lat_col]), float(r[lng_col])) for _, r in target_df.iterrows()]
        points = [origin] + coords
        if len(points) < 2:
            st.warning("경로 계산 대상이 부족합니다."); return
        D = build_distance_matrix(points)
        if len(points) == 2:
            opt_idx = [0, 1]
        else:
            nn = nearest_neighbor_route(D, start_idx=0)
            opt_idx = two_opt(nn, D)
        visit_idx = [i for i in opt_idx if i != 0]
        base_km = route_length(opt_idx, D) if len(opt_idx) > 1 else 0.0
        ordered_boxes = serialize_route(visit_idx, ids)  # <-- 보정된 함수 사용
        st.success(f"기본 경로 총거리: {base_km:.2f} km")

        final_order = ordered_boxes[:]
        if ai_help:
            if not OPENAI_KEY:
                st.warning("OPENAI_API_KEY가 설정되어 있지 않습니다. Settings→Secrets에 추가하세요.")
            else:
                df_sub = target_df.set_index(target_df[box_col].astype(str)).loc[ordered_boxes].reset_index(drop=True)
                ai_order = ai_refine_route(OPENAI_KEY, ordered_boxes, df_sub, box_col, meta_cols, base_km, start_label)
                if ai_order:
                    idx_map = {bid:i+1 for i,bid in enumerate(ids)}  # points 인덱스
                    try:
                        ai_visit_idx = [idx_map[b] for b in ai_order]
                        ai_route = [0] + ai_visit_idx
                        ai_km = route_length(ai_route, D) if len(ai_route) > 1 else 0.0
                        st.info(f"AI 재조정 경로 총거리: {ai_km:.2f} km")
                        if ai_km <= base_km * 1.2 + 1e-6:
                            final_order = ai_order; st.success("AI 경로 적용됨(거리 20% 이내).")
                        else:
                            st.warning("AI 경로가 거리 제한 초과. 기본 경로 유지.")
                    except Exception as e:
                        st.warning(f"AI 경로 검증 실패: {e}")

        # 지도 표출
        id_to_coord = {str(r[box_col]): (float(r[lat_col]), float(r[lng_col])) for _, r in target_df.iterrows()}
        order_pos = [(b, id_to_coord[b]) for b in final_order]
        route_map = folium.Map(location=[origin[0], origin[1]], zoom_start=14, tiles="OpenStreetMap")
        folium.Marker(location=[origin[0], origin[1]], icon=folium.Icon(color="green", icon="play"), tooltip="출발지").add_to(route_map)
        poly = [origin]
        for i, (bid, (la, lo)) in enumerate(order_pos, start=1):
            folium.Marker(location=[la, lo], tooltip=f"{i}. 박스 {bid}",
                          icon=folium.DivIcon(html=f"<div style='font-size:12px;color:#000;background:#fff;border:1px solid #333;border-radius:10px;padding:2px 6px;'>{i}</div>")).add_to(route_map)
            poly.append((la, lo))
        folium.PolyLine(locations=poly, weight=5, opacity=0.8).add_to(route_map)
        st.markdown("**최종 경로 지도**"); st_folium(route_map, width=None, height=600)

        # 결과 표
        dist_rows, total_km, prev = [], 0.0, origin
        for i, (bid, (la, lo)) in enumerate(order_pos, start=1):
            seg = _haversine(prev[0], prev[1], la, lo); total_km += seg
            dist_rows.append({"순번": i, "박스번호": bid, "세그먼트거리(km)": round(seg,2), "누적거리(km)": round(total_km,2)})
            prev = (la, lo)
        res_df = pd.DataFrame(dist_rows)
        st.dataframe(res_df, use_container_width=True); st.metric("총 주행거리", f"{total_km:.2f} km")

        # 내보내기
        colx, coly = st.columns(2)
        with colx:
            csv = res_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("경로 CSV 다운로드", data=csv,
                               file_name=f"수거경로_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        with coly:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as w: res_df.to_excel(w, index=False, sheet_name="수거경로")
            st.download_button("경로 엑셀 다운로드", data=buf.getvalue(),
                               file_name=f"수거경로_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ---------- 거리표 옵션 ----------
    show_distances = show_distances and filtered_boxes and len(filtered_boxes) > 1
    if show_distances:
        st.subheader("📏 박스 간 거리")
        distance_df = df_clean[df_clean[box_col].astype(str).isin(filtered_boxes)]
        if len(distance_df) > 1:
            distances = []
            idxs = list(distance_df.index)
            for a_i, i in enumerate(idxs):
                for b_i, j in enumerate(idxs):
                    if a_i < b_i:
                        r1, r2 = distance_df.loc[i], distance_df.loc[j]
                        d = calculate_distance(float(r1[lat_col]), float(r1[lng_col]), float(r2[lat_col]), float(r2[lng_col]))
                        distances.append({'박스1': str(r1[box_col]), '박스2': str(r2[box_col]), '거리(km)': round(d,2)})
            if distances:
                ddf = pd.DataFrame(distances).sort_values('거리(km)')
                st.dataframe(ddf, use_container_width=True)
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("평균 거리", f"{ddf['거리(km)'].mean():.2f} km")
                with c2: st.metric("최단 거리", f"{ddf['거리(km)'].min():.2f} km")
                with c3: st.metric("최장 거리", f"{ddf['거리(km)'].max():.2f} km")

    # ---------- 원본 데이터 내보내기 ----------
    st.subheader("💾 데이터 내보내기")
    export_df = df_clean[df_clean[box_col].astype(str).isin(filtered_boxes)] if filtered_boxes else df_clean
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
        st.metric("표시된 박스 수", len(export_df))
        if '상태' in export_df.columns:
            normal_count = len(export_df[export_df['상태'] == '정상'])
            st.metric("정상 상태 박스", normal_count)
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
