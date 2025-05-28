# Nama         : Clarisya Adeline
# NPM          : 140810230017
# Tanggal Buat : 27/05/2025
# Deskripsi    : Dominant Color Palette dengan Light/Dark Mode

import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import cv2  # Untuk konversi ruang warna dan operasi gambar

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="🎨 Dominant Color Picker",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_theme_css(is_dark_mode):
    """Generate CSS untuk menyesuaikan tema Light dan Dark"""
    if is_dark_mode:
        return """
        <style>
        [data-testid="stFileUploader"] button {
            background-color: #4b6cb7 !important;
            color: white !important;
            border: none !important;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            cursor: pointer;
        }

        [data-testid="stFileUploader"] button:hover {
            background-color: #182848 !important;
        }
    
        [data-testid="stFileUploader"] section {
            background-color: #222 !important;
            border: 2px dashed #555 !important;
            color: white !important;
            border-radius: 8px;
            padding: 1rem;
        }

        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
        }

        [data-testid="stSidebar"] {
            background-color: #1e1e2f !important;
        }

        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* File uploader styling */
        [data-testid="stFileUploader"] {
            background-color: #000000 !important;
            border: 1px solid #555 !important;
            border-radius: 10px;
        }

        [data-testid="stFileUploader"] * {
            color: white !important;
        }

        /* Tombol */
        .stButton > button {
            background-color: #4b6cb7 !important;
            color: white !important;
            border: none;
            border-radius: 5px;
        }

        .stButton > button:hover {
            background-color: #182848 !important;
        }

        [data-testid="stDownloadButton"] button {
            background-color: #4b6cb7 !important;
            color: white !important;
            border-radius: 5px !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
        }
        [data-testid="stDownloadButton"] button:hover {
            background-color: #182848 !important;
            color: white !important;
        }

        /* Umum */
        h1, h2, h3, h4, h5, h6, .stMarkdown {
            color: white !important;
        }

        .block-container {
            background-color: rgba(30, 30, 50, 0.95);
            border-radius: 15px;
            padding: 2rem;
        }

        .color-box {
                display: inline-block;
                width: 120px;
                height: 80px;
                margin: 5px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
                border: 2px solid #444;
            }
            
            .color-info {
                text-align: center;
                font-weight: bold;
                margin-top: 5px;
                padding: 5px;
                background: rgba(50, 50, 70, 0.8);
                color: #ffffff;
                border-radius: 5px;
            }
            
            .title {
                text-align: center;
                color: #ffffff;
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }
            
            .subtitle {
                text-align: center;
                color: #b0b0b0;
                font-size: 1.2rem;
                margin-bottom: 2rem;
            }
            
            .info-card {
                background: rgba(40, 40, 60, 0.8);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .stDataFrame {
                background: rgba(30, 30, 50, 0.9);
            }
            
            h1, h2, h3, h4, h5, h6 {
                color: #ffffff !important;
            }
            
            .stMarkdown {
                color: #ffffff;
            }
            
            .theme-toggle {
                background: rgba(40, 40, 60, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
            }
        </style>
        """
    else:
        return """
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #000000;
        }

        [data-testid="stFileUploader"] section {
            background-color: #2c3e50 !important;
            border: 2px dashed #ccc !important;
            color: #333 !important;
        }        

        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
        }

        [data-testid="stSidebar"] * {
            color: #000000 !important;
        }

        /* File uploader styling */
        [data-testid="stFileUploader"] {
            background-color: #2c3e50 !important;
            border: 1px solid #ccc !important;
            border-radius: 10px;
        }

        [data-testid="stFileUploader"] * {
            color: #ffffff !important;
        }

        [data-testid="stFileUploader"] button {
            background-color: #e0e0e0 !important;
            color: #2c3e50 !important;
            border: 1px solid rgba(0, 0, 0, 0.2);
            border-radius: 5px;
            padding: 0.5rem 1rem;
            cursor: pointer;
        }

        [data-testid="stFileUploader"] button:hover {
                background-color: #c7c7c7 !important;
        }

        /* Tombol */
        .stButton > button {
            background-color: #adc7e0 !important;
            color: #2c3e50 !important;
            border: 1px solid rgba(0, 0, 0, 0.2);
            border-radius: 5px;
        }

        .stButton > button:hover {
            background-color: #d5d5d5 !important;
        }

        [data-testid="stDownloadButton"] button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 5px !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
        }
        [data-testid="stDownloadButton"] button:hover {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        /* Umum */
        h1, h2, h3, h4, h5, h6, .stMarkdown {
            color: #2c3e50 !important;
        }

        .block-container {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
        }

        .color-box {
                display: inline-block;
                width: 120px;
                height: 80px;
                margin: 5px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                border: 2px solid white;
            }
            
            .color-info {
                text-align: center;
                font-weight: bold;
                margin-top: 5px;
                padding: 5px;
                background: rgba(255, 255, 255, 0.8);
                color: #2c3e50;
                border-radius: 5px;
            }
            
            .title {
                text-align: center;
                color: #2c3e50;
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .subtitle {
                text-align: center;
                color: #34495e;
                font-size: 1.2rem;
                margin-bottom: 2rem;
            }
            
            .info-card {
                background: rgba(255, 255, 255, 0.9);
                color: #2c3e50;
                border: 1px solid rgba(0, 0, 0, 0.1);
            }
            
            h1, h2, h3, h4, h5, h6 {
                color: #2c3e50 !important;
            }
            
            .theme-toggle {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
            }
        </style>
        """
def style_df(df, dark_mode):
    if dark_mode:
        return df.style.set_table_styles([
            {'selector': 'thead th', 'props': [('color', 'white'), ('background-color', '#1e1e2f')]},
            {'selector': 'tbody td', 'props': [('color', 'white'), ('background-color', '#2a2a40')]},
            {'selector': 'tbody tr:hover', 'props': [('background-color', '#444466')]},
        ]).set_properties(**{'border-color': '#555'})
    else:
        return df.style.set_table_styles([
            {'selector': 'thead th', 'props': [('color', '#2c3e50'), ('background-color', '#e9ecef')]},
            {'selector': 'tbody td', 'props': [('color', '#2c3e50'), ('background-color', 'white')]},
            {'selector': 'tbody tr:hover', 'props': [('background-color', '#f0f0f0')]},
        ]).set_properties(**{'border-color': '#ccc'})

def rgb_to_hex(rgb):
    """Konversi warna dari format RGB (array) ke hex string"""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def weighted_sampling_pixels(img_array):
    """
    Fungsi untuk melakukan weighted sampling piksel gambar berdasarkan saturasi dan brightness.
    Tujuannya agar warna cerah (saturasi & brightness tinggi) punya pengaruh lebih besar di clustering.
    """
    # Konversi gambar dari RGB ke BGR untuk OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # Konversi dari BGR ke HSV agar bisa dapatkan saturasi dan brightness
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Flatten array piksel RGB dan HSV ke bentuk 2D (jumlah_piksel x 3)
    pixels_rgb = img_array.reshape(-1, 3)
    pixels_hsv = img_hsv.reshape(-1, 3)
    
    saturation = pixels_hsv[:,1]  # Saturasi tiap piksel (0-255)
    brightness = pixels_hsv[:,2]  # Brightness (value) tiap piksel (0-255)
    
    # Hitung bobot sebagai hasil perkalian saturasi dan brightness (nilai 0-1)
    weights = (saturation / 255) * (brightness / 255)
    
    # Oversampling: Duplikasi piksel dengan bobot tinggi lebih banyak kali
    repeated_pixels = []
    for i, w in enumerate(weights):
        n = max(1, int(w * 10))  # Maksimal duplikat 10 kali
        repeated_pixels.extend([pixels_rgb[i]] * n)
    
    # Kembalikan array piksel hasil sampling ulang
    return np.array(repeated_pixels)

def extract_dominant_colors(image, k=5):
    """
    Ekstrak warna dominan menggunakan KMeans clustering.
    Proses:
    - Weighted sampling piksel berdasarkan saturasi & brightness
    - Konversi piksel hasil sampling ke ruang warna Lab (CIELAB)
    - Jalankan KMeans clustering di Lab space
    - Konversi centroid cluster kembali ke RGB untuk visualisasi
    - Hitung persentase piksel tiap cluster
    """
    img_array = np.array(image)
    weighted_pixels = weighted_sampling_pixels(img_array)
    
    # Konversi weighted sampled pixels dari RGB ke BGR, lalu ke Lab
    weighted_bgr = cv2.cvtColor(weighted_pixels.reshape(-1,1,3), cv2.COLOR_RGB2BGR).reshape(-1,3)
    weighted_lab = cv2.cvtColor(weighted_bgr.reshape(-1,1,3), cv2.COLOR_BGR2Lab).reshape(-1,3)
    
    # Jalankan KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(weighted_lab)
    
    # Ambil centroid cluster di Lab space
    centers_lab = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_
    
    # Hitung persentase tiap cluster
    label_counts = np.bincount(labels)
    percentages = (label_counts / len(labels)) * 100
    
    # Konversi centroid dari Lab ke BGR lalu ke RGB
    centers_lab_reshaped = centers_lab.reshape(-1,1,3)
    centers_bgr = cv2.cvtColor(centers_lab_reshaped, cv2.COLOR_Lab2BGR).reshape(-1,3)
    centers_rgb = cv2.cvtColor(centers_bgr.reshape(-1,1,3), cv2.COLOR_BGR2RGB).reshape(-1,3)
    
    # Urutkan warna berdasarkan persentase terbanyak
    sorted_indices = np.argsort(percentages)[::-1]
    colors = centers_rgb[sorted_indices]
    percentages = percentages[sorted_indices]
    
    return colors, percentages

def create_color_palette(colors, percentages, is_dark_mode=False):
    """
    Membuat visualisasi distribusi warna dalam bentuk bar chart horizontal
    dengan styling sesuai tema
    """
    # Set style matplotlib sesuai tema
    if is_dark_mode:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 1, figsize=(12, 3), facecolor='#1e1e32')
        ax.set_facecolor('#1e1e32')
        text_color = 'white'
        edge_color = '#444'
    else:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 3), facecolor='white')
        ax.set_facecolor('white')
        text_color = 'black'
        edge_color = 'white'
    
    x_pos = 0
    for i, (color, percentage) in enumerate(zip(colors, percentages)):
        ax.barh(0, percentage, left=x_pos, height=0.8, 
                color=color/255, edgecolor=edge_color, linewidth=2)
        # ax.text(x_pos + percentage/2, 0, f'{percentage:.1f}%', 
        #         ha='center', va='center', fontweight='bold', 
        #         color='white' if sum(color) < 384 else 'black')
        x_pos += percentage
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Persentase Warna (%)', fontsize=12, fontweight='bold', color=text_color)
    ax.set_title('Distribusi Warna Dominan', fontsize=14, fontweight='bold', pad=20, color=text_color)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(text_color)
    ax.tick_params(colors=text_color)
    
    plt.tight_layout()
    return fig

def create_color_palette_image(colors, percentages, is_dark_mode=False):
    """
    Membuat gambar color palette dalam bentuk kotak warna horizontal
    yang bisa didownload sebagai PNG
    """
    # Set warna background dan text berdasarkan tema
    bg_color = '#1e1e32' if is_dark_mode else '#ffffff'
    text_color = 'white' if is_dark_mode else 'black'
    
    # Ukuran gambar
    fig_width = 12
    fig_height = 10
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # Buat kotak warna
    num_colors = len(colors)
    box_width = 1.0 / num_colors
    
    for i, (color, percentage) in enumerate(zip(colors, percentages)):
        # Gambar kotak warna
        rect = plt.Rectangle((i * box_width, 0.4), box_width, 0.4, 
                           facecolor=color/255, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        
        # Tambahkan teks informasi warna
        hex_color = rgb_to_hex(color)
        rgb_text = f"RGB({color[0]}, {color[1]}, {color[2]})"
        percentage_text = f"{percentage:.1f}%"
        
        # Posisi teks di tengah kotak
        text_x = i * box_width + box_width/2
        
        # Hex code
        ax.text(text_x, 0.85, hex_color, ha='center', va='center', 
                fontsize=9, fontweight='bold', color=text_color)
        
        # RGB values
        ax.text(text_x, 0.25, rgb_text, ha='center', va='center', 
                fontsize=8, color=text_color)
        
        # Percentage
        ax.text(text_x, 0.1, percentage_text, ha='center', va='center', 
                fontsize=8, fontweight='bold', color=text_color)
        
        # Percentage di dalam kotak warna
        text_color_inside = 'white' if sum(color) < 384 else 'black'
        ax.text(text_x, 0.6, percentage_text, ha='center', va='center', 
                fontsize=9, fontweight='bold', color=text_color_inside)
    
    # Set limits dan remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Tambahkan judul
    ax.text(0.5, 0.95, 'Color Palette', ha='center', va='center', 
            fontsize=16, fontweight='bold', color=text_color)
    
    plt.tight_layout()
    return fig

def main():
    # Inisialisasi state untuk theme
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Apply CSS berdasarkan tema yang dipilih
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    # Judul dan subjudul aplikasi
    st.markdown('<h1 class="title">🎨 Dominant Color Picker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ekstrak 5 warna dominan dari gambar menggunakan K-Means Clustering dengan weighted sampling</p>', unsafe_allow_html=True)
    
    # Sidebar pengaturan
    with st.sidebar:
        st.header("⚙️ Dashboard")
        st.markdown("---")
        
        # Theme Toggle
        
        # st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
        st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
        st.subheader("🎭 Tema Aplikasi")
        
        col_theme1, col_theme2 = st.columns(2)
        with col_theme1:
            if st.button("☀️ Light Mode", use_container_width=True, 
                        type="primary" if not st.session_state.dark_mode else "secondary"):
                st.session_state.dark_mode = False
                st.rerun()
                            
        with col_theme2:
            if st.button("🌙 Dark Mode", use_container_width=True,
                        type="primary" if st.session_state.dark_mode else "secondary"):
                st.session_state.dark_mode = True
                st.rerun()
        
        current_theme = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode"
        st.markdown(f"**Tema aktif:** {current_theme}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Upload gambar
        uploaded_file = st.file_uploader(
            "📁 Pilih gambar",
            type=['png', 'jpg', 'jpeg'],
            help="Upload gambar dalam format PNG, JPG, atau JPEG"
        )
        
        # Slider untuk memilih jumlah cluster (warna dominan)
        num_colors = st.slider(
            "🎨 Jumlah warna dominan",
            min_value=3,
            max_value=10,
            value=5,
            help="Pilih berapa banyak warna dominan yang ingin diekstrak"
        )
        
        # Penjelasan metode
        st.markdown("---")
        st.markdown("### 📚 Tentang Metode")
        st.markdown("""
        K-Means clustering dengan weighted sampling piksel berdasarkan saturasi dan brightness,
        agar warna cerah lebih berpengaruh dan warna minor tetap muncul walau jumlah cluster terbatas.
        """)
    
    # Layout dua kolom: kiri gambar asli, kanan warna dominan
    col1, col2 = st.columns([1,1])
    
    if uploaded_file is not None:
        # Buka dan pastikan mode RGB
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        with col1:
            st.subheader("📸 Gambar Asli")
            st.image(image, use_container_width=True, caption="Gambar yang diupload")
            st.markdown("---")
            
            # Info card dengan styling sesuai tema
            info_style = "info-card" 
            # st.markdown(f'<div class="{info_style}" style="padding: 1rem; border-radius: 10px; margin: 1rem 0;">', unsafe_allow_html=True)
            st.markdown("**📊 Informasi Gambar:**")
            st.write(f"• Ukuran: {image.size[0]} × {image.size[1]} pixels")
            st.write(f"• Mode: {image.mode}")
            st.write(f"• Format: {uploaded_file.type}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("🎨 Warna Dominan")
            # Proses ekstraksi warna dengan spinner loading
            with st.spinner("🔄 Mengekstrak warna dominan..."):
                colors, percentages = extract_dominant_colors(image, num_colors)
                color_html = ""
                # Tampilkan kotak warna + info hex, rgb, dan persentase
                for i, (color, percentage) in enumerate(zip(colors, percentages)):
                    hex_color = rgb_to_hex(color)
                    color_html += f"""
                    <div style="display: inline-block; margin: 10px; text-align: center;">
                        <div class="color-box" style="background-color: {hex_color};"></div>
                        <div class="color-info">
                            <div style="font-size: 0.9em; font-weight: bold;">{hex_color}</div>
                            <div style="font-size: 0.8em; opacity: 0.8;">RGB({color[0]}, {color[1]}, {color[2]})</div>
                            <div style="font-size: 0.8em; opacity: 0.8;">{percentage:.1f}%</div>
                        </div>
                    </div>
                    """
                st.markdown(color_html, unsafe_allow_html=True)
        
        # Visualisasi grafik distribusi warna
        st.markdown("---")
        st.subheader("📊 Distribusi Warna")
        fig = create_color_palette(colors, percentages, st.session_state.dark_mode)
        st.pyplot(fig)
        
        # Tabel detail warna
        st.subheader("📋 Detail Warna")
        color_data = []
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            color_data.append({
                'Urutan': i+1,
                'Hex': rgb_to_hex(color),
                'RGB': f"({color[0]}, {color[1]}, {color[2]})",
                'Persentase': f"{percentage:.2f}%"
            })
        df = pd.DataFrame(color_data)
        st.dataframe(style_df(df, st.session_state.dark_mode), use_container_width=True)
        
        # Download tombol color palette dan data CSV
        st.markdown("---")
        st.subheader("💾 Download Hasil")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # Buat color palette image untuk download
            palette_fig = create_color_palette_image(colors, percentages, st.session_state.dark_mode)
            buf = BytesIO()
            palette_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                              facecolor=palette_fig.get_facecolor())
            buf.seek(0)
            st.download_button(
                label="🎨 Download Color Palette",
                data=buf.getvalue(),
                file_name=f"color_palette_{uploaded_file.name.split('.')[0]}.png",
                mime="image/png"
            )
        with col_dl2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download Data CSV",
                data=csv,
                file_name=f"color_data_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv"
            )
    else:
        # Tampilan awal jika belum upload gambar
        welcome_style = "info-card"
        st.markdown(f"""
        <div class="{welcome_style}" style="text-align: center; padding: 3rem; border-radius: 15px; margin: 2rem 0;">
            <h3>🖼️ Upload Gambar untuk Memulai</h3>
            <p>Pilih gambar dari sidebar untuk mengekstrak warna dominan menggunakan K-Means clustering dengan weighted sampling</p>
            <br>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <h4>🎯 Fitur Utama</h4>
                    <ul style="text-align: left; display: inline-block;">
                        <li>Ekstraksi 3-10 warna dominan</li>
                        <li>Weighted sampling piksel berdasarkan saturasi dan brightness</li>
                        <li>Visualisasi distribusi warna</li>
                        <li>Format Hex dan RGB</li>
                        <li>Download hasil analisis</li>
                        <li>🆕 Light/Dark Mode Toggle</li>
                    </ul>
                </div>
                <div style="text-align: center;">
                    <h4>🔬 Teknologi</h4>
                    <ul style="text-align: left; display: inline-block;">
                        <li>K-Means Clustering</li>
                        <li>OpenCV (cv2)</li>
                        <li>PIL (Python Imaging)</li>
                        <li>Matplotlib</li>
                        <li>Streamlit</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer kecil di bagian bawah halaman
    st.markdown("---")
    theme_emoji = "🌙" if st.session_state.dark_mode else "☀️"
    st.markdown(f"""
    <div style="text-align: center; opacity: 0.7; padding: 1rem;">
        <p>🎨 <strong>Dominant Color Picker</strong> {theme_emoji} - Dibuat dengan ❤️ oleh Clarisya Adeline</p>
        <p><em>Upload gambar dan temukan warna-warna dominan yang tersembunyi!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
