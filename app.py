# app.py
# 실행: pip install streamlit folium streamlit-folium scikit-learn openpyxl numpy requests
# 실행: streamlit run app.py

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
import os, io, math, json, requests
from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime

st.set_page_config(page_title="암롤박스 위치 조회 시스템", page_icon="📍", layout="wide")

# -------------------- 유틸 --------------------
def _hav(lat1, lng1, lat2, lng2):
    R = 6371.0
    a1, a2 = math.radians(lat1), math.radians(lat2)
    d1, d2 = math.radians(lat2-lat1), math.radians(lng2-lng1)
    a = math.sin(d1/2)**2 + math.cos(a1)*math.cos(a2)*math.sin(d2/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def build_D(points: List[Tuple[float,float]]) -> np.ndarray:
    n=len(points); D=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            d=_hav(points[i][0],points[i][1],points[j][0],points[j][1])
            D[i,j]=D[j,i]=d
    return D

def nn_route(D: np.ndarray, start: int=0)->List[int]:
    n=D.shape[0]; U=set(range(n)); U.remove(start); r=[start]; c=start
    while U:
        nx=min(U,key=lambda j:D[c,j]); r.append(nx); U.remove(nx); c=nx
    return r

def two_opt(route: List[int], D: np.ndarray, itmax: int=2000)->List[int]:
    r=route[:]; n=len(r); it=0; improved=True
    while improved and it<itmax:
        improved=False; it+=1
        for i in range(1,n-2):
            for k in range(i+1,n-1):
                a,b=r[i-1],r[i]; c,d=r[k],r[k+1]
                if D[a,b]+D[c,d] > D[a,c]+D[b,d]:
                    r[i:k+1]=reversed(r[i:k+1]); improved=True
    return r

def route_len(route: List[int], D: np.ndarray)->float:
    return sum(D[route[i], route[i+1]] for i in range(len(route)-1))

def serialize_route(idx_list: List[int], ids: List[str])->List[str]:
    # points 인덱스(0=origin, 1..n=박스) → ids(0..n-1)
    return [ids[i-1] for i in idx_list]

def safe_float(x)->Optional[float]:
    try: return float(x)
    except: return None

# -------------------- 데이터 --------------------
@st.cache_data
def load_excel_data(file_path: str, sheet_name: str = "좌표정보") -> Optional[pd.DataFrame]:
    try:
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        except ValueError:
            xl = pd.ExcelFile(file_path, engine='openpyxl')
            if xl.sheet_names:
                st.warning(f"'{sheet_name}' 시트 없음 → '{xl.sheet_names[0]}' 사용")
                return pd.read_excel(file_path, sheet_name=xl.sheet_names[0], engine='openpyxl')
            st.error("읽을 수 있는 시트가 없습니다."); return None
    except Exception as e:
        st.error(f"엑셀 로드 오류: {e}")
        return None

def detect_columns(df: pd.DataFrame)->dict:
    found={'box':None,'lat':None,'lng':None}
    for c in df.columns:
        s=str(c).strip()
        if s=='박스번호': found['box']=c
        elif s=='위도(DD)': found['lat']=c
        elif s=='경도(DD)': found['lng']=c
    if not all(found.values()):
        for c in df.columns:
            low=str(c).lower()
            if found['box'] is None and any(k in low for k in ['박스번호','box','박스','번호','number','id','no']): found['box']=c
            if found['lat'] is None and any(k in low for k in ['위도(dd)','위도','lat','latitude','y']): found['lat']=c
            if found['lng'] is None and any(k in low for k in ['경도(dd)','경도','lng','lon','longitude','x']): found['lng']=c
    return found

def parse_boxes(text: str)->List[str]:
    if not text or not text.strip(): return []
    return [t.strip() for t in text.split(',') if t.strip()]

def perform_kmeans(df, lat_col, lng_col, box_col, k):
    coords=df[[lat_col,lng_col]].values
    km=KMeans(n_clusters=k, random_state=42, n_init='auto')
    lab=km.fit_predict(coords)
    dfx=df.copy(); dfx['cluster']=lab
    info={}
    for i in range(k):
        sub=dfx[dfx['cluster']==i]; c=km.cluster_centers_[i]
        info[i]={'center_lat':float(c[0]),'center_lng':float(c[1]),'box_count':len(sub),
                 'box_numbers':sub[box_col].astype(str).tolist(),'boxes_data':sub}
    return dfx, info

# -------------------- 지도 --------------------
def draw_map(df, box_col, lat_col, lng_col, filtered=None, use_cluster=True, style="OpenStreetMap", k=None, selected=None):
    data=df[df[box_col].astype(str).isin(filtered)] if filtered else df.copy()
    center=(34.926111, 127.764722)
    tiles, attr = ("OpenStreetMap", None)
    if style=="위성지도": tiles, attr=("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}","Esri")
    elif style=="지형지도": tiles, attr=("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}","Esri")
    elif style=="CartoDB": tiles, attr=("CartoDB positron", None)
    m=folium.Map(location=center, zoom_start=14, tiles=tiles, attr=attr) if attr else folium.Map(location=center, zoom_start=14, tiles=tiles)
    cluster_info=None
    if use_cluster and len(data)>5:
        if k and k>0 and k<len(data):
            data, cluster_info = perform_kmeans(data, lat_col, lng_col, box_col, k)
            container=m
        else:
            container=MarkerCluster(showCoverageOnHover=True, zoomToBoundsOnClick=True).add_to(m)
    else:
        container=m
    if cluster_info and selected is not None and len(cluster_info)>1 and 'cluster' in data.columns:
        ids=[int(s.split()[-1])-1 for s in selected]
        data=data[data['cluster'].isin(ids)]
    colors=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD','#98D8C8','#F7DC6F','#BB8FCE','#85C1E9']
    for _,r in data.iterrows():
        try:
            lat, lng = float(r[lat_col]), float(r[lng_col]); bid=str(r[box_col])
            color = colors[int(r['cluster'])%len(colors)] if 'cluster'in r else '#3498DB'
            folium.Circle(location=[lat,lng], radius=15,
                          popup=folium.Popup(f"박스 {bid}", max_width=250),
                          tooltip=f"📦 {bid}", color=color, fill=True, fill_color=color,
                          weight=10, fill_opacity=0.7, opacity=0.9).add_to(container)
        except: continue
    return m, cluster_info

# -------------------- AI 보조 --------------------
def ai_refine_route(api_key: str, candidate_order: List[str], df_sub: pd.DataFrame,
                    box_col: str, extra_cols: List[str], base_km: float, start_label: str)->Optional[List[str]]:
    rows=[]
    for _,r in df_sub.iterrows():
        item={"box":str(r[box_col])}
        for c in extra_cols:
            v=r.get(c,None)
            if pd.notna(v): item[c]=v if isinstance(v,(int,float)) else str(v)[:60]
        rows.append(item)
    system=("You are a logistics optimizer. Output ONLY JSON with key 'order' as list of box IDs in visiting order. "
            "Prefer higher priority, status=점검/수리 first. Do not increase distance >20% vs baseline.")
    user={"start_label":start_label,"baseline_km":round(base_km,3),"candidate_order":candidate_order,
          "metadata_columns":extra_cols,"items":rows}
    try:
        h={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
        p={"model":"gpt-4o-mini","messages":[{"role":"system","content":system},
           {"role":"user","content":json.dumps(user, ensure_ascii=False)}],
           "temperature":0.1,"response_format":{"type":"json_object"}}
        r=requests.post("https://api.openai.com/v1/chat/completions",headers=h,json=p,timeout=30)
        if r.status_code!=200:
            st.warning(f"OpenAI 오류: {r.status_code} {r.text[:150]}"); return None
        content=r.json()["choices"][0]["message"]["content"]
        data=json.loads(content)
        out=data.get("order") if isinstance(data,dict) else None
        if not isinstance(out,list): st.warning("AI 응답 형식 오류"); return None
        out=[str(x) for x in out]
        cand=set(candidate_order)
        return [x for x in out if x in cand]+[x for x in candidate_order if x not in out]
    except Exception as e:
        st.warning(f"AI 경로 재조정 실패: {e}")
        return None

# -------------------- 메인 --------------------
def main():
    if "route_state" not in st.session_state:
        st.session_state["route_state"]=None   # 마지막 경로 결과 보관

    st.title("📍 암롤박스 위치 조회 시스템")
    st.markdown("---")

    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY","")  # Settings→Secrets에 {"OPENAI_API_KEY":"sk-..."} 저장

    excel_file="box_locations.xlsx"
    github_raw="https://raw.githubusercontent.com/YunSuGeun-kor/posco/main/box_locations.xlsx"

    if not os.path.exists(excel_file):
        with st.spinner("Github에서 엑셀 파일을 불러오는 중..."):
            try:
                r=requests.get(github_raw, timeout=15)
                if r.status_code==200:
                    df=load_excel_data(io.BytesIO(r.content),"좌표정보")
                else:
                    df=None
            except: df=None
        if df is None:
            st.error("로컬/원격 모두 실패. 엑셀 업로드 필요.")
            up=st.file_uploader("엑셀 파일 선택", type=['xlsx','xls'])
            if up is not None:
                with open(excel_file,"wb") as f: f.write(up.getbuffer())
                st.success("업로드 완료"); st.rerun()
            st.stop()
    else:
        df=load_excel_data(excel_file,"좌표정보")
        if df is None: st.stop()

    st.success(f"✅ 데이터 로딩: {len(df)}행")

    det=detect_columns(df)
    c1,c2,c3=st.columns(3)
    with c1:
        box_col=st.selectbox("박스 번호 컬럼", df.columns.tolist(),
                             index=df.columns.tolist().index(det['box']) if det['box'] else 0)
    with c2:
        lat_col=st.selectbox("위도 컬럼", df.columns.tolist(),
                             index=df.columns.tolist().index(det['lat']) if det['lat'] else 0)
    with c3:
        lng_col=st.selectbox("경도 컬럼", df.columns.tolist(),
                             index=df.columns.tolist().index(det['lng']) if det['lng'] else 0)

    if not all([box_col,lat_col,lng_col]): st.error("모든 컬럼 선택 필요"); st.stop()

    dfc=df.dropna(subset=[lat_col,lng_col]).copy()
    dfc[lat_col]=dfc[lat_col].apply(safe_float); dfc[lng_col]=dfc[lng_col].apply(safe_float)
    dfc=dfc.dropna(subset=[lat_col,lng_col])
    if len(dfc)==0: st.error("유효 좌표 없음"); st.stop()

    st.subheader("🔍 박스 필터링")
    fc1,fc2=st.columns([3,1])
    with fc1: box_input=st.text_input("조회할 박스 번호(쉼표 구분)", placeholder="예: 101, 205, 333")
    with fc2: st.button("🔍 조회", type="primary")
    filtered=parse_boxes(box_input) if box_input else None

    if filtered:
        exist=dfc[box_col].astype(str).tolist()
        miss=[b for b in filtered if b not in exist]
        st.info(f"필터링: {len(filtered)}개"); 
        if miss: st.warning(f"데이터 없음: {', '.join(miss)}")
    else:
        st.info(f"전체 표시: {len(dfc)}개")

    st.subheader("⚙️ 지도 옵션")
    m1,m2,m3,m4=st.columns(4)
    with m1: map_type=st.selectbox("지도 종류", ["일반지도","위성지도","지형지도","CartoDB"], index=0)
    with m2: use_cluster=st.checkbox("마커 클러스터링", value=True)
    with m3:
        mode=st.radio("클러스터링 모드", ["자동","수동"], index=0, disabled=not use_cluster)
        k=st.number_input("클러스터 개수", 1, 50, 3) if (use_cluster and mode=="수동") else None
    with m4: show_dist=st.checkbox("박스 간 거리표", value=False)

    _, info = draw_map(dfc, box_col, lat_col, lng_col, filtered, use_cluster, map_type, k)
    selected=None
    if use_cluster and info and len(info)>1:
        labels=[f"클러스터 {i+1}" for i in range(len(info))]
        if 'selected_clusters' not in st.session_state: st.session_state['selected_clusters']=labels.copy()
        b1,b2=st.columns(2)
        with b1:
            if st.button("전체 선택"): st.session_state['selected_clusters']=labels.copy()
        with b2:
            if st.button("전체 취소"): st.session_state['selected_clusters']=[]
        selected=st.multiselect("표시할 클러스터", labels, default=st.session_state['selected_clusters'], key="selected_clusters")

    base_map, _ = draw_map(dfc, box_col, lat_col, lng_col, filtered, use_cluster, map_type, k, selected)
    st.subheader("🗺️ 위치 지도")
    st_folium(base_map, width=None, height=600, key="base_map")

    # -------------------- 수거 경로 최적화 --------------------
    st.subheader("🛣️ 수거 경로 최적화")
    target = dfc[dfc[box_col].astype(str).isin(filtered)] if filtered else dfc.copy()
    if len(target)==0:
        st.info("대상 박스가 없습니다.")
        return

    start_mode=st.radio("출발지", ["소각로(기본)","임의 좌표 입력","특정 박스에서 시작"], horizontal=True)
    default_origin=(34.926111, 127.764722)
    if start_mode=="임의 좌표 입력":
        oc1,oc2=st.columns(2)
        with oc1: s_lat=st.number_input("출발 위도", value=default_origin[0], format="%.6f")
        with oc2: s_lng=st.number_input("출발 경도", value=default_origin[1], format="%.6f")
        origin=(s_lat, s_lng); start_label="custom_origin"
    elif start_mode=="특정 박스에서 시작":
        s_box=st.selectbox("시작 박스", options=target[box_col].astype(str).tolist())
        row=target[target[box_col].astype(str)==s_box].iloc[0]
        origin=(float(row[lat_col]), float(row[lng_col])); start_label=f"box:{s_box}"
    else:
        origin=default_origin; start_label="incinerator"

    ai_help=st.checkbox("AI 보조 의사결정(우선순위/상태 반영)", value=False)
    meta_cols=[]
    if ai_help:
        cand=[c for c in target.columns if c not in [box_col,lat_col,lng_col,'cluster']]
        meta_cols=st.multiselect("의사결정 컬럼", options=cand, default=[c for c in cand if c in ["우선순위","상태","적재량"]])

    col_run, col_clear = st.columns([1,1])
    run = col_run.button("🚚 경로 계산", type="primary")
    clear = col_clear.button("🧹 경로 초기화")

    if clear:
        st.session_state["route_state"]=None

    if run:
        ids=target[box_col].astype(str).tolist()
        coords=[(float(r[lat_col]), float(r[lng_col])) for _,r in target.iterrows()]
        points=[origin]+coords
        D=build_D(points)
        if len(points)==2:
            opt=[0,1]
        else:
            opt=two_opt(nn_route(D,0), D)
        visit=[i for i in opt if i!=0]
        base_km=route_len(opt, D) if len(opt)>1 else 0.0
        ordered=serialize_route(visit, ids)

        final_order=ordered[:]
        ai_km=None
        if ai_help and OPENAI_KEY:
            df_sub=target.set_index(target[box_col].astype(str)).loc[ordered].reset_index(drop=True)
            ai_order=ai_refine_route(OPENAI_KEY, ordered, df_sub, box_col, meta_cols, base_km, start_label)
            if ai_order:
                idx_map={bid:i+1 for i,bid in enumerate(ids)}
                try:
                    ai_idx=[idx_map[b] for b in ai_order]
                    ai_route=[0]+ai_idx
                    ai_km=route_len(ai_route, D)
                    if ai_km <= base_km*1.2 + 1e-6:
                        final_order=ai_order
                except: pass
        # 세션에 저장 → rerun 후에도 유지
        id2coord={str(r[box_col]):(float(r[lat_col]), float(r[lng_col])) for _,r in target.iterrows()}
        st.session_state["route_state"]={
            "origin": origin,
            "order": final_order,
            "ordered_coords": [id2coord[b] for b in final_order],
            "base_km": base_km,
            "ai_km": ai_km
        }

    # 세션에 저장된 경로가 있으면 항상 렌더(사라짐 방지)
    rs=st.session_state.get("route_state")
    if rs:
        rmap=folium.Map(location=[rs["origin"][0], rs["origin"][1]], zoom_start=14, tiles="OpenStreetMap")
        folium.Marker(location=[rs["origin"][0], rs["origin"][1]],
                      icon=folium.Icon(color="green", icon="play"),
                      tooltip="출발지").add_to(rmap)
        poly=[rs["origin"]]
        for i,(la,lo) in enumerate(rs["ordered_coords"], start=1):
            folium.Marker(location=[la,lo],
                          tooltip=f"{i}",
                          icon=folium.DivIcon(html=f"<div style='font-size:12px;color:#000;background:#fff;border:1px solid #333;border-radius:10px;padding:2px 6px;'>{i}</div>")).add_to(rmap)
            poly.append((la,lo))
        folium.PolyLine(locations=poly, weight=5, opacity=0.85).add_to(rmap)
        st.markdown("**최종 경로 지도**")
        st_folium(rmap, width=None, height=600, key="route_map")

        # 표와 지표
        rows=[]; tot=0.0; prev=rs["origin"]
        for i,(la,lo) in enumerate(rs["ordered_coords"], start=1):
            seg=_hav(prev[0],prev[1],la,lo); tot+=seg
            rows.append({"순번":i,"박스번호":rs["order"][i-1],"세그먼트(km)":round(seg,2),"누적(km)":round(tot,2)})
            prev=(la,lo)
        out_df=pd.DataFrame(rows)
        st.dataframe(out_df, use_container_width=True)
        m1,m2=st.columns(2)
        with m1: st.metric("기본 총거리", f"{rs['base_km']:.2f} km")
        with m2:
            st.metric("최종 총거리", f"{tot:.2f} km")
        c1,c2=st.columns(2)
        with c1:
            csv=out_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("경로 CSV 다운로드", data=csv,
                               file_name=f"수거경로_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        with c2:
            buf=io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as w: out_df.to_excel(w, index=False, sheet_name="수거경로")
            st.download_button("경로 엑셀 다운로드", data=buf.getvalue(),
                               file_name=f"수거경로_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # 거리표
    if show_dist and filtered and len(filtered)>1:
        st.subheader("📏 박스 간 거리")
        df_d = dfc[dfc[box_col].astype(str).isin(filtered)]
        dist=[]
        idx=list(df_d.index)
        for a_i,i in enumerate(idx):
            for b_i,j in enumerate(idx):
                if a_i<b_i:
                    r1,r2=df_d.loc[i], df_d.loc[j]
                    d=_hav(float(r1[lat_col]), float(r1[lng_col]), float(r2[lat_col]), float(r2[lng_col]))
                    dist.append({'박스1':str(r1[box_col]),'박스2':str(r2[box_col]),'거리(km)':round(d,2)})
        if dist:
            ddf=pd.DataFrame(dist).sort_values('거리(km)')
            st.dataframe(ddf, use_container_width=True)
            c1,c2,c3=st.columns(3)
            with c1: st.metric("평균", f"{ddf['거리(km)'].mean():.2f} km")
            with c2: st.metric("최단", f"{ddf['거리(km)'].min():.2f} km")
            with c3: st.metric("최장", f"{ddf['거리(km)'].max():.2f} km")

    # 원본/필터 데이터 내보내기
    st.subheader("💾 데이터 내보내기")
    exp=dfc[dfc[box_col].astype(str).isin(filtered)] if filtered else dfc
    ex1,ex2=st.columns(2)
    with ex1:
        if st.button("📊 Excel로 내보내기"):
            b=io.BytesIO()
            with pd.ExcelWriter(b, engine='openpyxl') as w: exp.to_excel(w, index=False, sheet_name='암롤박스_데이터')
            st.download_button("💾 Excel 다운로드", data=b.getvalue(),
                               file_name=f"암롤박스_위치정보_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with ex2:
        if st.button("📄 CSV로 내보내기"):
            c=exp.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("💾 CSV 다운로드", data=c,
                               file_name=f"암롤박스_위치정보_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")

if __name__ == "__main__":
    main()
