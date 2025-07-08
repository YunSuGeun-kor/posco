import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
import os
import math
import io
from typing import List, Optional
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="암롤박스 위치 조회 시스템",
    page_icon="📍",
    layout="wide"
)

@st.cache_data
def load_excel_data(file_path: str, sheet_name: str = "좌표정보") -> Optional[pd.DataFrame]:
    """
    Load Excel data with error handling, specifically looking for coordinate sheet
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to load (default: "좌표정보")
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        # First, try to load the specific sheet
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            st.success(f"'{sheet_name}' 시트에서 데이터를 로드했습니다.")
            return df
        except ValueError:
            # If the specific sheet doesn't exist, try to find available sheets
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            available_sheets = excel_file.sheet_names
            st.warning(f"'{sheet_name}' 시트를 찾을 수 없습니다. 사용 가능한 시트: {', '.join(available_sheets)}")
            
            # Try to load the first sheet
            if available_sheets:
                df = pd.read_excel(file_path, sheet_name=available_sheets[0], engine='openpyxl')
                st.info(f"'{available_sheets[0]}' 시트에서 데이터를 로드했습니다.")
                return df
            else:
                st.error("읽을 수 있는 시트가 없습니다.")
                return None
                
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {str(e)}")
        # Try to provide more specific error information
        if "not a zip file" in str(e).lower():
            st.error("파일이 올바른 Excel 형식이 아닙니다. .xlsx 또는 .xls 파일인지 확인해주세요.")
        elif "no such file" in str(e).lower():
            st.error("파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None

def detect_columns(df: pd.DataFrame) -> dict:
    """
    Automatically detect column names for box number, latitude, and longitude
    Specifically designed for the Excel structure:
    - 박스번호 (box number)
    - 위도(DD) (latitude)
    - 경도(DD) (longitude)
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with detected column names
    """
    detected = {'box': None, 'lat': None, 'lng': None}
    
    # Check for exact matches first
    for col in df.columns:
        col_clean = str(col).strip()
        
        # Exact matches for the specified structure
        if col_clean == '박스번호':
            detected['box'] = col
        elif col_clean == '위도(DD)':
            detected['lat'] = col
        elif col_clean == '경도(DD)':
            detected['lng'] = col
    
    # If exact matches not found, try pattern matching
    if not all(detected.values()):
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Detect box column
            if detected['box'] is None:
                box_patterns = ['박스번호', 'box', '박스', '번호', 'number', 'id', 'no']
                for pattern in box_patterns:
                    if pattern in col_lower:
                        detected['box'] = col
                        break
            
            # Detect latitude column
            if detected['lat'] is None:
                lat_patterns = ['위도(dd)', '위도', 'lat', 'latitude', 'y']
                for pattern in lat_patterns:
                    if pattern in col_lower:
                        detected['lat'] = col
                        break
            
            # Detect longitude column
            if detected['lng'] is None:
                lng_patterns = ['경도(dd)', '경도', 'lng', 'lon', 'longitude', 'x']
                for pattern in lng_patterns:
                    if pattern in col_lower:
                        detected['lng'] = col
                        break
    
    return detected

def parse_box_numbers(input_text: str) -> List[str]:
    """
    Parse comma-separated box numbers from input text
    
    Args:
        input_text: User input string
        
    Returns:
        List of box numbers as strings
    """
    if not input_text.strip():
        return []
    
    # Split by comma and clean up
    box_numbers = [num.strip() for num in input_text.split(',') if num.strip()]
    return box_numbers

def perform_kmeans_clustering(df: pd.DataFrame, lat_col: str, lng_col: str, box_col: str, 
                             n_clusters: int) -> tuple[pd.DataFrame, dict]:
    """
    Perform K-means clustering on box locations
    
    Args:
        df: DataFrame with box data
        lat_col: Column name for latitude
        lng_col: Column name for longitude
        box_col: Column name for box numbers
        n_clusters: Number of clusters to create
        
    Returns:
        Tuple of (DataFrame with cluster labels, dictionary with cluster info)
    """
    # Prepare coordinates for clustering
    coordinates = df[[lat_col, lng_col]].values
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(coordinates)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Create cluster information dictionary
    cluster_info = {}
    for i in range(n_clusters):
        cluster_boxes = df_clustered[df_clustered['cluster'] == i]
        cluster_center = kmeans.cluster_centers_[i]
        
        cluster_info[i] = {
            'center_lat': float(cluster_center[0]),
            'center_lng': float(cluster_center[1]),
            'box_count': len(cluster_boxes),
            'box_numbers': cluster_boxes[box_col].astype(str).tolist(),
            'boxes_data': cluster_boxes
        }
    
    return df_clustered, cluster_info

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate distance between two points using Haversine formula
    
    Args:
        lat1, lng1: First point coordinates
        lat2, lng2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2) +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lng / 2) * math.sin(delta_lng / 2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

def create_map(df: pd.DataFrame, box_col: str, lat_col: str, lng_col: str, 
               filtered_boxes: Optional[List[str]] = None, use_clustering: bool = True, 
               map_style: str = "OpenStreetMap", cluster_count: Optional[int] = None,
               selected_clusters: Optional[list] = None) -> tuple[folium.Map, Optional[dict]]:
    """
    Create folium map with box locations and clustering
    
    Args:
        df: DataFrame with box data
        box_col: Column name for box numbers
        lat_col: Column name for latitude
        lng_col: Column name for longitude
        filtered_boxes: List of box numbers to display (None for all)
        use_clustering: Whether to use marker clustering
        map_style: Map tile style ('OpenStreetMap', 'Satellite', 'CartoDB positron', etc.)
        cluster_count: Custom cluster count (None for automatic clustering)
        selected_clusters: List of selected clusters to filter
        
    Returns:
        Tuple of (Folium map object, cluster information dictionary)
    """
    # Filter data if specific boxes are requested
    if filtered_boxes:
        # Convert box numbers to string for comparison
        df_filtered = df[df[box_col].astype(str).isin(filtered_boxes)]
    else:
        df_filtered = df.copy()
    
    # Set map center to fixed location (광양제철소 소각로)
    center_lat, center_lng = 34.926111, 127.764722
    
    # Initialize cluster_info
    cluster_info = None
    
    # Define tile options based on map style
    if map_style == "위성지도":
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attr = "Esri"
    elif map_style == "지형지도":
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
        attr = "Esri"
    elif map_style == "CartoDB":
        tiles = "CartoDB positron"
        attr = None
    else:  # Default to OpenStreetMap
        tiles = "OpenStreetMap"
        attr = None

    # Create map
    if attr:
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=14,
            tiles=tiles,
            attr=attr
        )
    else:
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=14,
            tiles=tiles
        )
    
    # Handle clustering based on user preferences
    if use_clustering and len(df_filtered) > 5:
        if cluster_count and cluster_count > 0:
            # Use K-means clustering for fixed cluster count
            if cluster_count >= len(df_filtered):
                # If cluster count >= marker count, no clustering needed
                container = m
            else:
                # Perform K-means clustering
                df_clustered, cluster_info = perform_kmeans_clustering(
                    df_filtered, lat_col, lng_col, box_col, cluster_count
                )
                df_filtered = df_clustered
                container = m  # No folium clustering, we'll use K-means results
        else:
            # Automatic clustering with folium MarkerCluster
            marker_cluster = MarkerCluster(
                showCoverageOnHover=True,
                zoomToBoundsOnClick=True
            ).add_to(m)
            container = marker_cluster
    else:
        container = m
    
    # Define improved cluster colors for better visual distinction
    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                     '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                     '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2',
                     '#FAD7A0', '#A9DFBF', '#F9E79F', '#D5A6BD']
    
    # 클러스터링이 활성화되고, 클러스터가 2개 이상이며, selected_clusters가 지정된 경우 해당 클러스터만 필터링
    if cluster_info and selected_clusters is not None and len(cluster_info) > 1:
        selected_ids = [int(s.split()[-1]) - 1 for s in selected_clusters]
        if isinstance(df_filtered, pd.DataFrame) and 'cluster' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['cluster'].isin(selected_ids)]

    # Add markers for each box
    for idx, row in df_filtered.iterrows():
        try:
            lat = float(row[lat_col])
            lng = float(row[lng_col])
            box_num = str(row[box_col])
            
            # Create detailed popup with additional info
            popup_html = f"""
            <div style="font-family: 'Noto Sans KR', sans-serif; font-size: 14px;">
                <h4 style="margin: 5px 0; color: #2E86AB;">📦 박스 {box_num}</h4>
                <p style="margin: 3px 0;"><b>위도:</b> {lat:.6f}</p>
                <p style="margin: 3px 0;"><b>경도:</b> {lng:.6f}</p>
            """
            
            # Add cluster information if available
            if cluster_info and 'cluster' in row:
                cluster_id = int(row['cluster'])
                popup_html += f"<p style='margin: 3px 0;'><b>클러스터:</b> {cluster_id + 1}</p>"
            
            # Add additional columns if they exist
            additional_cols = [col for col in df_filtered.columns 
                             if col not in [box_col, lat_col, lng_col, 'cluster']]
            
            for col in additional_cols[:3]:  # Limit to 3 additional columns
                try:
                    value = row[col]
                    if pd.notna(value):
                        popup_html += f"<p style='margin: 3px 0;'><b>{col}:</b> {value}</p>"
                except KeyError:
                    continue
            
            popup_html += "</div>"
            
            # Choose marker color
            if cluster_info and 'cluster' in row:
                # Use cluster-based coloring for K-means
                cluster_id = int(row['cluster'])
                marker_color = cluster_colors[cluster_id % len(cluster_colors)]
            elif '상태' in df_filtered.columns and pd.notna(row['상태']):
                # Use status-based coloring with improved colors
                status = str(row['상태']).lower()
                if '정상' in status:
                    marker_color = '#2ECC71'  # Bright green
                elif '점검' in status:
                    marker_color = '#F39C12'  # Orange
                elif '수리' in status:
                    marker_color = '#E74C3C'  # Red
                else:
                    marker_color = '#3498DB'  # Blue
            else:
                marker_color = '#FF00FF'  # Default red #E74C3C
            
            # 마커 추가 분기: 클러스터링(군집) vs 개별/군집1개
            is_single_cluster = False
            if cluster_info and len(cluster_info) == 1:
                is_single_cluster = True
            # MarkerCluster(자동)에서 마커 수가 1개인 경우도 포함
            if (use_clustering and len(df_filtered) > 5 and (cluster_info or isinstance(container, MarkerCluster)) and not is_single_cluster):
                # 클러스터링(군집, 2개 이상) 마커: 기존 CircleMarker 유지
                folium.Circle(
                    location=[lat, lng],
                    radius=15,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"📦 박스 {box_num}",
                    fill=True,
                    color=marker_color,
                    fill_color=marker_color,
                    weight=11.5,
                    fill_opacity=0.7,
                    opacity=0.9
                ).add_to(container)
            else:
                # 개별 박스 또는 군집 1개: folium.Circle (지도상 실제 반지름, 예: 15m)
                folium.Circle(
                    location=[lat, lng],
                    radius=15,  # 단위: 미터
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"📦 박스 {box_num}",
                    color=marker_color,
                    fill=True,
                    fill_color=marker_color,
                    weight=11.5,
                    fill_opacity=0.7,
                    opacity=0.9
                ).add_to(container)
            
        except (ValueError, TypeError) as e:
            st.warning(f"박스 {row[box_col]}의 좌표 정보가 올바르지 않습니다: {e}")
            continue
    
    # Add legend for K-means clusters if cluster_info exists
    if cluster_info:
        # Create legend HTML
        legend_html = '''
        <div style="position: fixed; 
                    top: 50%; left: 10px; width: 180px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);
                    transform: translateY(-50%);
                    font-family: 'Noto Sans KR', Arial, sans-serif;
                    ">
        <h4 style="margin: 0 0 10px 0; text-align: center; color: #333; font-size: 16px;">클러스터 범례</h4>
        '''
        
        # Add each cluster to the legend
        for cluster_id in sorted(cluster_info.keys()):
            cluster_data = cluster_info[cluster_id]
            color = cluster_colors[cluster_id % len(cluster_colors)]
            
            # Use the actual hex colors from cluster_colors
            css_color = color  # color is already in hex format
            box_count = cluster_data['box_count']
            
            legend_html += f'''
            <div style="margin: 8px 0; padding: 5px; background-color: #f9f9f9; border-radius: 3px;">
                <div style="display: flex; align-items: center; margin-bottom: 2px;">
                    <div style="width: 18px; height: 18px; background-color: {css_color}; 
                               border: 2px solid #333; border-radius: 50%; margin-right: 8px;
                               display: inline-block;"></div>
                    <span style="color: #333; font-weight: bold; font-size: 13px;">클러스터 {cluster_id + 1}</span>
                </div>
                <div style="margin-left: 26px;">
                    <span style="color: #666; font-size: 11px;">박스 {box_count}개</span>
                </div>
            </div>
            '''
        
        legend_html += '</div>'
        
        # Add legend to map
        legend_element = folium.Element(legend_html)
        m.get_root().html.add_child(legend_element)
    
    return m, cluster_info

def main():
    """Main application function"""
    
    st.title("📍 암롤박스 위치 조회 시스템")
    st.markdown("---")
    
    # File upload or use existing file # GitHub에 업로드된 파일을 상대 경로로 불러오기
    excel_file = "암롤박스위치정보.xlsx"
    github_raw_url = "https://raw.githubusercontent.com/YunSuGeun-kor/posco/main/암롤박스위치정보.xlsx"
    
    # Check if file exists
    if not os.path.exists(excel_file):
        #github_raw_url에서 시도
        try:
            with st.spinner("Github에서 엑셀 파일을 불러오는 중입니다..."):
                df = load_excel_data(github_raw_url,"좌표정보")
            if df is None:
                raise Exception("GitHub에서 파일을 불러오지 못했습니다.")
        except Exception as e:
            st.error(f"엑셀 파일 '{excel_file}'을 찾을 수 없고, GitHub 저장소에서도 불러오지 못했습니다. 파일을 업로드해주세요.")
            st.info("엑셀 파일을 업로드해주세요.")
            uploaded_file = st.file_uploader(
                "엑셀 파일 선택",
                type=['xlsx', 'xls'],
                help="암롤박스 위치 정보가 포함된 엑셀 파일을 업로드하세요."
            )
        
            if uploaded_file is not None:
                # Save uploaded file
                with open(excel_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("파일이 업로드되었습니다!")
                st.rerun()
            else:
                st.stop()
    else:
        #파일이 있으면 기존대로 진행
        # Load data from the specific sheet
        with st.spinner("데이터를 로딩중입니다..."):
            df = load_excel_data(excel_file, "좌표정보")
    
        if df is None:
            st.stop()
    
    # Display basic information
    st.success(f"✅ 데이터 로딩 완료: {len(df)}개의 레코드")
    
    # Detect columns
    detected_cols = detect_columns(df)
    
    # Column selection
    st.subheader("🔧 컬럼 설정")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        box_col = st.selectbox(
            "박스 번호 컬럼",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(detected_cols['box']) if detected_cols['box'] else 0
        )
    
    with col2:
        lat_col = st.selectbox(
            "위도 컬럼",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(detected_cols['lat']) if detected_cols['lat'] else 0
        )
    
    with col3:
        lng_col = st.selectbox(
            "경도 컬럼",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(detected_cols['lng']) if detected_cols['lng'] else 0
        )
    
    # Validate selected columns
    if not all([box_col, lat_col, lng_col]):
        st.error("모든 컬럼을 선택해주세요.")
        st.stop()
    
    # Remove rows with missing coordinates
    df_clean = df.dropna(subset=[lat_col, lng_col])
    
    if len(df_clean) == 0:
        st.error("유효한 좌표 데이터가 없습니다.")
        st.stop()
    
    st.info(f"유효한 좌표를 가진 박스: {len(df_clean)}개")
    
    # Filter section
    st.subheader("🔍 박스 필터링")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        box_input = st.text_input(
            "조회할 박스 번호 입력 (쉼표로 구분)",
            placeholder="예: 101, 205, 333",
            help="박스 번호를 쉼표로 구분하여 입력하세요. 비워두면 모든 박스가 표시됩니다."
        )
    
    with col2:
        search_button = st.button("🔍 조회", type="primary")
    
    # Parse input
    filtered_boxes = parse_box_numbers(box_input) if box_input else None
    
    # Display current filter status
    if filtered_boxes:
        st.info(f"필터링된 박스: {', '.join(filtered_boxes)} ({len(filtered_boxes)}개)")
        
        # Check which boxes exist in data
        existing_boxes = df_clean[box_col].astype(str).tolist()
        missing_boxes = [box for box in filtered_boxes if box not in existing_boxes]
        
        if missing_boxes:
            st.warning(f"다음 박스는 데이터에 없습니다: {', '.join(missing_boxes)}")
    else:
        st.info(f"전체 박스 표시: {len(df_clean)}개")
    
    # Map options
    st.subheader("⚙️ 지도 옵션")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        map_type = st.selectbox(
            "지도 종류",
            options=["일반지도", "위성지도", "지형지도", "CartoDB"],
            index=0,
            help="지도 배경을 선택하세요"
        )
    
    with col2:
        use_clustering = st.checkbox("마커 클러스터링 사용", value=True, 
                                   help="많은 마커가 있을 때 클러스터링으로 성능 향상")
    
    with col3:
        cluster_mode = st.radio(
            "클러스터링 모드",
            options=["자동", "수동"],
            index=0,
            help="자동: 자동으로 최적 클러스터 생성, 수동: 직접 클러스터 개수 설정",
            disabled=not use_clustering
        )
        
        if use_clustering and cluster_mode == "수동":
            cluster_count = st.number_input(
                "클러스터 개수",
                min_value=1,
                max_value=50,
                value=3,
                help="원하는 클러스터 개수를 입력하세요"
            )
        else:
            cluster_count = None
    
    with col4:
        show_distances = st.checkbox("박스 간 거리 계산", value=False,
                                   help="선택된 박스들 간의 거리를 계산하여 표시")
    
    # 1. cluster_info 추출용 임시 지도 생성
    _, cluster_info = create_map(
        df_clean, box_col, lat_col, lng_col,
        filtered_boxes, use_clustering, map_type, cluster_count
    )

    # 2. 클러스터별 선택 UI
    selected_clusters = None
    if use_clustering and cluster_info and len(cluster_info) > 1:
        cluster_labels = [f"클러스터 {i+1}" for i in range(len(cluster_info))]
        if 'selected_clusters' not in st.session_state:
            st.session_state['selected_clusters'] = cluster_labels.copy()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("전체 선택"):
                st.session_state['selected_clusters'] = cluster_labels.copy()
        with col2:
            if st.button("전체 취소"):
                st.session_state['selected_clusters'] = []
        selected_clusters = st.multiselect(
            "표시할 클러스터 선택",
            options=cluster_labels,
            default=st.session_state['selected_clusters'],
            key="selected_clusters"
        )

    # 3. 실제 지도 생성 (선택된 클러스터 반영)
    folium_map, cluster_info = create_map(
        df_clean, box_col, lat_col, lng_col,
        filtered_boxes, use_clustering, map_type, cluster_count,
        selected_clusters=selected_clusters
    )

    # Create and display map
    st.subheader("🗺️ 위치 지도")
    
    try:
        with st.spinner("지도를 생성중입니다..."):
            # Display map
            map_data = st_folium(
                folium_map,
                width=2000,
                height=1000,
                returned_objects=["last_object_clicked"]
            )
            
            # Display clicked marker info
            if map_data["last_object_clicked"]:
                clicked_info = map_data["last_object_clicked"]
                st.success(f"선택된 위치: 위도 {clicked_info['lat']:.6f}, 경도 {clicked_info['lng']:.6f}")
        
    except Exception as e:
        st.error(f"지도 생성 중 오류가 발생했습니다: {str(e)}")
    
    # Distance calculation section
    if show_distances and filtered_boxes and len(filtered_boxes) > 1:
        st.subheader("📏 박스 간 거리")
        
        # Get filtered data for distance calculation
        distance_df = df_clean[df_clean[box_col].astype(str).isin(filtered_boxes)]
        
        if len(distance_df) > 1:
            distances = []
            indices = list(distance_df.index)
            for i_idx, i in enumerate(indices):
                for j_idx, j in enumerate(indices):
                    if i_idx < j_idx:  # Avoid duplicate pairs
                        row1 = distance_df.loc[i]
                        row2 = distance_df.loc[j]
                        box1 = str(row1[box_col])
                        box2 = str(row2[box_col])
                        lat1, lng1 = float(row1[lat_col]), float(row1[lng_col])
                        lat2, lng2 = float(row2[lat_col]), float(row2[lng_col])
                        
                        dist = calculate_distance(lat1, lng1, lat2, lng2)
                        distances.append({
                            '박스1': box1,
                            '박스2': box2,
                            '거리(km)': round(dist, 2)
                        })
            
            if distances:
                distance_df_display = pd.DataFrame(distances)
                distance_df_display = distance_df_display.sort_values('거리(km)')
                st.dataframe(distance_df_display, use_container_width=True)
                
                # Show summary statistics
                avg_distance = distance_df_display['거리(km)'].mean()
                min_distance = distance_df_display['거리(km)'].min()
                max_distance = distance_df_display['거리(km)'].max()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("평균 거리", f"{avg_distance:.2f} km")
                with col2:
                    st.metric("최단 거리", f"{min_distance:.2f} km")
                with col3:
                    st.metric("최장 거리", f"{max_distance:.2f} km")
    
    # Export functionality
    st.subheader("💾 데이터 내보내기")
    
    # Prepare export data
    if filtered_boxes:
        export_df = df_clean[df_clean[box_col].astype(str).isin(filtered_boxes)]
        export_title = f"필터링된 박스 데이터 ({len(export_df)}개)"
    else:
        export_df = df_clean
        export_title = f"전체 박스 데이터 ({len(export_df)}개)"
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel export
        if st.button("📊 Excel 파일로 내보내기"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='암롤박스_데이터', index=False)
            
            st.download_button(
                label="💾 Excel 파일 다운로드",
                data=buffer.getvalue(),
                file_name=f"암롤박스_위치정보_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        # CSV export
        if st.button("📄 CSV 파일로 내보내기"):
            csv = export_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="💾 CSV 파일 다운로드",
                data=csv,
                file_name=f"암롤박스_위치정보_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Data preview
    with st.expander("📊 데이터 미리보기"):
        st.write(export_title)
        
        # Show additional statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("표시된 박스 수", len(export_df))
        with col2:
            if '상태' in export_df.columns:
                normal_count = len(export_df[export_df['상태'] == '정상'])
                st.metric("정상 상태 박스", normal_count)
        with col3:
            if len(export_df) > 0:
                lat_range = export_df[lat_col].max() - export_df[lat_col].min()
                st.metric("위도 범위", f"{lat_range:.4f}°")
        
        # Show cluster composition if K-means clustering was used
        if cluster_info:
            st.subheader("🎯 클러스터별 박스 구성")
            
            # Create tabs for each cluster
            cluster_tabs = st.tabs([f"클러스터 {i+1}" for i in range(len(cluster_info))])
            
            for i, tab in enumerate(cluster_tabs):
                with tab:
                    cluster_data = cluster_info[i]
                    
                    # Cluster summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("박스 개수", cluster_data['box_count'])
                    with col2:
                        st.metric("중심 위도", f"{cluster_data['center_lat']:.6f}")
                    with col3:
                        st.metric("중심 경도", f"{cluster_data['center_lng']:.6f}")
                    
                    # Box numbers in this cluster
                    st.write("**포함된 박스 번호:**")
                    box_numbers_str = ", ".join(cluster_data['box_numbers'])
                    st.text(box_numbers_str)
                    
                    # Detailed data table for this cluster
                    st.write("**클러스터 상세 데이터:**")
                    cluster_display_df = cluster_data['boxes_data'].drop(columns=['cluster'], errors='ignore')
                    st.dataframe(cluster_display_df, use_container_width=True)
            
            # Overall cluster summary table
            st.subheader("📈 클러스터 요약")
            summary_data = []
            for i, cluster_data in cluster_info.items():
                summary_data.append({
                    '클러스터': i + 1,
                    '박스 개수': cluster_data['box_count'],
                    '중심 위도': round(cluster_data['center_lat'], 6),
                    '중심 경도': round(cluster_data['center_lng'], 6),
                    '포함 박스': ', '.join(cluster_data['box_numbers'][:5]) + ('...' if len(cluster_data['box_numbers']) > 5 else '')
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        
        # Original data table
        st.subheader("📋 전체 데이터")
        st.dataframe(export_df, use_container_width=True)

if __name__ == "__main__":
    main()
