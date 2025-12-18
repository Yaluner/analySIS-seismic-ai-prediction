import streamlit as st
import pandas as pd
import folium
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from folium import Element
import os
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

# 1. Yapılandırma
st.set_page_config(layout="wide", page_title="analySIS")

# Hasar ve Senaryo Sistemi
def get_damage_scenario(intensity):
    
    # Mercalli Ölçeğine göre açıklama ve şiddetine göre renklendirme
    
    if intensity < 3.0:
        return "HİSSEDİLMEZ / ÇOK HAFİF", "Sadece hassas cihazlar veya üst katlardakiler hisseder.", "green", 10
    elif intensity < 4.0:
        return "HAFİF SARSINTI", "Eşyalar sallanır, uyuyanlar uyanır. Yapısal hasar beklenmez.", "#a3c400", 30
    elif intensity < 5.0:
        return "ORTA ŞİDDET (Hasar Sınırı)", "Sıva çatlakları, eşya devrilmeleri. Zayıf binalarda hafif hasar.", "orange", 50
    elif intensity < 6.0:
        return "GÜÇLÜ SARSINTI", "Duvar çatlakları, baca yıkılmaları. Eski binalarda orta hasar.", "#FF4500", 75
    elif intensity < 7.0:
        return "ÇOK GÜÇLÜ / YIKICI", "Binalarda yapısal hasar, duvar yıkılmaları riski.", "red", 90
    else:
        return "AFET DÜZEYİNDE YIKIM", "Ağır hasar ve göçme riski yüksek.", "#8B0000", 100

def calculate_damage_potential(magnitude, soil_factor):
    # Formula: Magnitude * (1 + (SoilFactor - 1) * 0.5)
    estimated_intensity = magnitude * (1 + (soil_factor - 1) * 0.5)
    damage_score = min(100, (estimated_intensity ** 2) * 1.5)
    return estimated_intensity, damage_score

# Renklendirme
def get_red_tone(magnitude):
    if magnitude < 3.0: return '#FFC0CB'
    elif magnitude < 4.0: return '#FA8072'
    elif magnitude < 5.0: return '#FF0000'
    else: return '#8B0000'
#Lejant
def add_legend(map_object):
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; z-index:9999; font-size:14px; background-color: white; color: black !important; opacity: 0.9; padding: 10px; border: 2px solid black; border-radius: 5px; font-family: Arial, sans-serif;">
        <h4 style="text-align: center; margin: 0; color: black;">LEJANT</h4>
        <div style="font-size:12px; line-height: 1.5; margin-top:5px;">
            <b>RİSK DURUMU:</b><br>
            <span style="background:green;width:10px;height:10px;display:inline-block;"></span> Düşük Risk<br>
            <span style="background:orange;width:10px;height:10px;display:inline-block;"></span> Orta Risk<br>
            <span style="background:red;width:10px;height:10px;display:inline-block;"></span> Yüksek Risk<br>
            <br>
            <b>GEÇMİŞ DEPREMLER:</b><br>
            <span style="background:#FA8072;width:10px;height:10px;display:inline-block;border:1px solid gray;"></span> Sismik Aktivite
        </div>
    </div>
    """
    map_object.get_root().html.add_child(Element(legend_html))

# Veri Yükleme
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='cp1254') 
        
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        
    if 'Mg' in df.columns:
        df['Mg'] = pd.to_numeric(df['Mg'], errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude', 'Depth', 'Mg'])
        
    return df

@st.cache_resource
def train_models_cached(df):
    features = ['Longitude', 'Latitude', 'Depth']
    if 'foreshock_count_7d' in df.columns and 'seismic_b_value' in df.columns:
        features += ['foreshock_count_7d', 'seismic_b_value']
    
    # Veri Temizleme
    noise_value = 0.556362
    clean_df = df[abs(df['seismic_b_value'] - noise_value) > 0.0001].copy()
    
    X = clean_df[features].fillna(0)
    y_class = (clean_df['Mg'] >= 4.0).astype(int)
    y_reg = clean_df['Mg']
    
    # Split
    X_train, X_test, y_c_train, y_c_test, y_r_train, y_r_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Risk Gruplandırma
    rf_class = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_class.fit(X_train, y_c_train)
    
    xgb_class = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, 
        use_label_encoder=False, eval_metric='logloss', random_state=42
    )
    xgb_class.fit(X_train, y_c_train)
    
    # Şiddet Gerilemesi
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_r_train)
    
    xgb_reg = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
    xgb_reg.fit(X_train, y_r_train)
    
    # Ölçeklendirme
    rf_acc = accuracy_score(y_c_test, rf_class.predict(X_test))
    xgb_acc = accuracy_score(y_c_test, xgb_class.predict(X_test))
    
    rf_mae = mean_absolute_error(y_r_test, rf_reg.predict(X_test))
    xgb_mae = mean_absolute_error(y_r_test, xgb_reg.predict(X_test))
    
    rf_r2 = r2_score(y_r_test, rf_reg.predict(X_test))
    xgb_r2 = r2_score(y_r_test, xgb_reg.predict(X_test))
    
    models = { 'rf_class': rf_class, 'xgb_class': xgb_class, 'rf_reg': rf_reg, 'xgb_reg': xgb_reg }
    metrics = { 'rf_acc': rf_acc, 'xgb_acc': xgb_acc, 'rf_mae': rf_mae, 'xgb_mae': xgb_mae, 'rf_r2': rf_r2, 'xgb_r2': xgb_r2 }
    
    return models, metrics, features, X_train

# Canlı Ölçeklendirme motoru
def calculate_live_metrics(lat, lon, df):
    if 'Date' not in df.columns: return 0, 0.0
    latest_date = df['Date'].max()
    start_date = latest_date - timedelta(days=7)
    recent_df = df[(df['Date'] >= start_date) & (df['Date'] <= latest_date)]
    if len(recent_df) == 0: return 0, 0.0
    R = 6371 
    phi1, phi2 = np.radians(lat), np.radians(recent_df['Latitude'])
    dphi = np.radians(recent_df['Latitude'] - lat)
    dlambda = np.radians(recent_df['Longitude'] - lon)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = R * c
    nearby_events = recent_df[distances <= 50]
    count = len(nearby_events)
    if count >= 10:
        mags = nearby_events['Mg'].values
        mean_mag = np.mean(mags)
        min_mag = np.min(mags)
        if mean_mag == min_mag: b_val = 0.0
        else: b_val = 0.4343 / (mean_mag - min_mag)
    else: b_val = 1.0 
    return count, b_val

# Dosya Yükleyici
def load_locations_from_file(file_path):
    locations = {}
    if not os.path.exists(file_path): return None, f"Dosya yok: {file_path}"
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1254']
    lines = []
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f: lines = f.readlines()
            break
        except UnicodeDecodeError: continue
    if not lines: return None, "Dosya okunamadı."
    current_city = None
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("#"):
            current_city = line.replace("#", "").strip()
            locations[current_city] = {}
        elif current_city: 
            parts = line.split('_')
            if len(parts) >= 3:
                name = parts[0].strip()
                lat = float(parts[1].replace(',', '.'))
                lon = float(parts[2].replace(',', '.'))
                locations[current_city][name] = (lat, lon)
    return locations, None

def load_soil_data(file_path):
    soil_map = {}
    if not os.path.exists(file_path): return {}
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1254']
    lines = []
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f: lines = f.readlines()
            break
        except UnicodeDecodeError: continue
    current_city = None
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("#"):
            current_city = line.replace("#", "").strip()
            soil_map[current_city] = {}
        elif current_city:
            parts = line.split('_')
            if len(parts) >= 2:
                district = parts[0].strip()
                try:
                    risk = float(parts[1])
                    soil_map[current_city][district] = risk
                except: continue
    return soil_map

def get_soil_factor(city, district, soil_map):
    default_risk = 1.3
    # 1. Direct match
    if city in soil_map:
        if district in soil_map[city]: return soil_map[city][district]
    
    try:
        city_lower = city.lower()
        district_lower = district.lower()
        for db_city in soil_map:
            if db_city.lower() == city_lower:
                for db_dist, val in soil_map[db_city].items():
                    if db_dist.lower() == district_lower: return val
    except: pass
    return default_risk

# Ana Uygulama
def main():
    st.title("AnalySIS Yapay Zeka Destekli Deprem Şiddet Ve Yıkım Tahmin Sistemi")
    
    if 'prediction_made' not in st.session_state:
        st.session_state['prediction_made'] = False
        st.session_state['risk_val'] = 0.0
        st.session_state['mag_val'] = 0.0
        st.session_state['soil_factor'] = 0.0
        st.session_state['damage_intensity'] = 0.0
        st.session_state['input_data'] = None
        st.session_state['input_coords'] = []
        st.session_state['selected_dist_name'] = ""
        st.session_state['calculated_metrics'] = (0, 0.0)
        st.session_state['is_simulation'] = False
        st.session_state['used_model'] = "XGBoost"

    # Pathler
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, "data_ready_for_ml.csv")
    loc_file = os.path.join(base_dir, "ilce_bilgi.txt")
    soil_file = os.path.join(base_dir, "zemin_bilgi.txt")
    
    df = load_data(data_file)
    if df is None: st.error(f"Veri yok: {data_file}"); st.stop()
    
    models, metrics, feature_names, X_train = train_models_cached(df)
    locations, error_msg = load_locations_from_file(loc_file)
    soil_map = load_soil_data(soil_file)

    tab1, tab2 = st.tabs(["Zemin & Hasar Analizi", "Model ve Parametreler"])
    
    # İlk sekme
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Konum ve Zemin")
            if locations:
                city = st.selectbox("Şehir:", list(locations.keys()))
                district = st.selectbox("İlçe:", list(locations[city].keys()))
                lat_def, lon_def = locations[city][district]
                soil_factor = get_soil_factor(city, district, soil_map)
                
                if soil_factor >= 2.0: st.warning(f"Zemin: ÇOK YUMUŞAK (Sıvılaşma Riski) - {soil_factor}x")
                elif soil_factor >= 1.5: st.info(f"Zemin: YUMUŞAK (Alüvyon) - {soil_factor}x")
                else: st.success(f"Zemin: SERT/KAYA - {soil_factor}x")
            else:
                city, district = "HATA", "Manuel"
                lat_def, lon_def = 38.0, 27.0
                soil_factor = 1.3
            
            depth = st.slider("Derinlik (km)", 1.0, 50.0, 7.0, 0.5)
            st.divider()
            model_choice = st.radio("Model Seçimi", ["XGBoost (Önerilen)", "Random Forest"], horizontal=True)
            st.divider()
            scenario_mode = st.checkbox("Senaryo Modu (Test)")
            
            sim_count, sim_b_val = 50, 0.6
            if scenario_mode:
                st.warning("Manuel Test Verisi Giriniz:")
                sim_count = st.slider("Öncü Şok (7 Gün)", 0, 200, 50)
                sim_b_val = st.slider("b-değeri", 0.1, 2.0, 0.6, 0.05)
            
            if st.button("ANALİZ ET VE HASAR HESAPLA", type="primary"):
                if scenario_mode: final_count, final_b_val = sim_count, sim_b_val; st.session_state['is_simulation'] = True
                else: final_count, final_b_val = calculate_live_metrics(lat_def, lon_def, df); st.session_state['is_simulation'] = False
                
                input_df = pd.DataFrame({'Longitude': [lon_def], 'Latitude': [lat_def], 'Depth': [depth],
                                       'foreshock_count_7d': [final_count], 'seismic_b_value': [final_b_val]})
                
                if "XGBoost" in model_choice:
                    risk_prob = models['xgb_class'].predict_proba(input_df)[0][1]
                    mag_pred = models['xgb_reg'].predict(input_df)[0]
                    used_model_name = "XGBoost"
                else:
                    risk_prob = models['rf_class'].predict_proba(input_df)[0][1]
                    mag_pred = models['rf_reg'].predict(input_df)[0]
                    used_model_name = "Random Forest"
                
                intensity, dmg_score = calculate_damage_potential(mag_pred, soil_factor)
                
                st.session_state['prediction_made'] = True
                st.session_state['risk_val'] = risk_prob
                st.session_state['mag_val'] = mag_pred
                st.session_state['soil_factor'] = soil_factor
                st.session_state['damage_intensity'] = intensity
                st.session_state['input_data'] = input_df 
                st.session_state['input_coords'] = [lat_def, lon_def]
                st.session_state['selected_dist_name'] = district
                st.session_state['calculated_metrics'] = (final_count, final_b_val)
                st.session_state['used_model'] = used_model_name
        
        with col2:
            if st.session_state['prediction_made']:
                risk = st.session_state['risk_val']
                mag = st.session_state['mag_val']
                intensity = st.session_state['damage_intensity']
                lat, lon = st.session_state['input_coords']
                
                if risk < 0.30: status, color = "DÜŞÜK", "green"
                elif risk < 0.60: status, color = "ORTA", "orange"
                else: status, color = "YÜKSEK", "red"
                
                st.info(f"Sonuç: **{st.session_state['selected_dist_name']}**")
                
                k1, k2, k3, k4 = st.columns(4)
                k1.metric(f"Risk ({st.session_state['used_model']})", f"%{risk*100:.1f}", delta=status, delta_color="inverse")
                k2.metric("Enerji (Mg)", f"{mag:.2f}")
                k3.metric("Zemin", f"x{st.session_state['soil_factor']}")
                k4.metric("Şiddet (MMI)", f"{intensity:.1f}", delta="Hasar Riski" if intensity>6.0 else "Normal")
                
                st.divider()
                st.markdown("### Tahmini Yıkım ve Hasar Analizi")
                dmg_title, dmg_desc, dmg_color, dmg_percent = get_damage_scenario(intensity)
                with st.container():
                    st.markdown(f"<h3 style='color:{dmg_color}; margin-bottom:0px;'>{dmg_title}</h3>", unsafe_allow_html=True)
                    st.progress(dmg_percent / 100)
                    st.info(f"**Beklenen Etki:** {dmg_desc}")

                m = folium.Map(location=[lat, lon], zoom_start=10)
                subset = df[df['Mg'] >= 3.0]
                for _, row in subset.iterrows():
                    folium.CircleMarker([row['Latitude'], row['Longitude']], radius=3,
                        color='gray', weight=0.5, fill=True, fill_color=get_red_tone(row['Mg']),
                        fill_opacity=0.6, popup=f"{row['Mg']} Mg").add_to(m)
                folium.CircleMarker([lat, lon], radius=30, color=color, fill=True, fill_color=color).add_to(m)
                add_legend(m)
                map_html = m.get_root().render()
                components.html(map_html, height=500)

    # 2. Sekme
    with tab2:
        st.header("Model Karşılaştırma & Karar Mekanizması")
        
        # 1. METRICS (Re-added Random Forest values)
        st.subheader("1. Performans Metrikleri")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Risk Tespiti (Doğruluk)**")
            st.metric("Random Forest", f"%{metrics['rf_acc']*100:.1f}")
            st.metric("XGBoost", f"%{metrics['xgb_acc']*100:.1f}", delta=f"{(metrics['xgb_acc']-metrics['rf_acc'])*100:.1f}%")
        with c2:
            st.markdown("**Büyüklük Hatası (MAE)**")
            st.metric("Random Forest", f"{metrics['rf_mae']:.2f}")
            st.metric("XGBoost", f"{metrics['xgb_mae']:.2f}", delta=f"{metrics['rf_mae']-metrics['xgb_mae']:.2f}", delta_color="inverse")
        with c3:
            st.markdown("**Açıklayıcılık (R²)**")
            st.metric("Random Forest", f"{metrics['rf_r2']:.2f}")
            st.metric("XGBoost", f"{metrics['xgb_r2']:.2f}", delta=f"{metrics['xgb_r2']-metrics['rf_r2']:.2f}")

        st.divider()
        st.subheader("2. Karar Gerekçeleri (SHAP Analizi)")

        if st.session_state['prediction_made'] and st.session_state['input_data'] is not None:
            col_xgb, col_rf = st.columns(2)
            
            with col_xgb:
                st.markdown("### XGBoost Görüşü")
                try:
                    with st.spinner("XGBoost hesaplanıyor..."):
                        explainer_xgb = shap.TreeExplainer(models['xgb_class'])
                        shap_values_xgb = explainer_xgb(st.session_state['input_data'])
                        fig_xgb, ax_xgb = plt.subplots(figsize=(6, 5))
                        shap.plots.waterfall(shap_values_xgb[0], show=False)
                        st.pyplot(fig_xgb)
                except Exception as e:
                    st.error(f"Grafik hatası: {e}")

            with col_rf:
                st.markdown("### Random Forest Görüşü")
                try:
                    with st.spinner("Random Forest hesaplanıyor..."):
                        explainer_rf = shap.TreeExplainer(models['rf_class'])
                        shap_values_rf = explainer_rf(st.session_state['input_data'])
                        
                        if isinstance(shap_values_rf, list): shap_val = shap_values_rf[1]
                        elif len(shap_values_rf.shape) == 3: shap_val = shap_values_rf[:, :, 1]
                        else: shap_val = shap_values_rf
                        
                        fig_rf, ax_rf = plt.subplots(figsize=(6, 5))
                        shap.plots.waterfall(shap_val[0], show=False)
                        st.pyplot(fig_rf)
                except Exception as e:
                     st.error(f"Grafik hatası (RF): {e}")
        else:
            st.warning("Karşılaştırmayı görmek için önce 'Analiz Et' butonuna basarak bir tahmin yapmalısınız.")

if __name__ == "__main__":
    main()


