# Nama         : Clarisya Adeline
# NPM          : 140810230017
# Tanggal Buat : 27/05/2025
# Deskripsi    : Dominant Color Palette

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

# CSS custom untuk styling halaman agar lebih menarik
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
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
</style>
""", unsafe_allow_html=True)

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

def create_color_palette(colors, percentages):
    """
    Membuat visualisasi distribusi warna dalam bentuk bar chart horizontal
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    x_pos = 0
    for i, (color, percentage) in enumerate(zip(colors, percentages)):
        ax.barh(0, percentage, left=x_pos, height=0.8, 
                color=color/255, edgecolor='white', linewidth=2)
        ax.text(x_pos + percentage/2, 0, f'{percentage:.1f}%', 
                ha='center', va='center', fontweight='bold', 
                color='white' if sum(color) < 384 else 'black')
        x_pos += percentage
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Persentase Warna (%)', fontsize=12, fontweight='bold')
    ax.set_title('Distribusi Warna Dominan', fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    return fig

def main():
    # Judul dan subjudul aplikasi
    st.markdown('<h1 class="title">🎨 Dominant Color Picker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ekstrak 5 warna dominan dari gambar menggunakan K-Means Clustering dengan weighted sampling</p>', unsafe_allow_html=True)
    
    # Sidebar pengaturan
    with st.sidebar:
        st.header("Dashboard")
        st.markdown("---")
        
        # Upload gambar
        uploaded_file = st.file_uploader(
            "Pilih gambar",
            type=['png', 'jpg', 'jpeg'],
            help="Upload gambar dalam format PNG, JPG, atau JPEG"
        )
        
        # Slider untuk memilih jumlah cluster (warna dominan)
        num_colors = st.slider(
            "Jumlah warna dominan",
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
            st.markdown("**Informasi Gambar:**")
            st.write(f"• Ukuran: {image.size[0]} × {image.size[1]} pixels")
            st.write(f"• Mode: {image.mode}")
            st.write(f"• Format: {uploaded_file.type}")
        
        with col2:
            st.subheader("🎨 Warna Dominan")
            # Proses ekstraksi warna dengan spinner loading
            with st.spinner("Mengekstrak warna dominan..."):
                colors, percentages = extract_dominant_colors(image, num_colors)
                color_html = ""
                # Tampilkan kotak warna + info hex, rgb, dan persentase
                for i, (color, percentage) in enumerate(zip(colors, percentages)):
                    hex_color = rgb_to_hex(color)
                    color_html += f"""
                    <div style="display: inline-block; margin: 10px; text-align: center;">
                        <div class="color-box" style="background-color: {hex_color};"></div>
                        <div class="color-info">
                            <div style="font-size: 0.9em;">{hex_color}</div>
                            <div style="font-size: 0.8em; color: #666;">RGB({color[0]}, {color[1]}, {color[2]})</div>
                            <div style="font-size: 0.8em; color: #666;">{percentage:.1f}%</div>
                        </div>
                    </div>
                    """
                st.markdown(color_html, unsafe_allow_html=True)
        
        # Visualisasi grafik distribusi warna
        st.markdown("---")
        st.subheader("📊 Distribusi Warna")
        fig = create_color_palette(colors, percentages)
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
        st.dataframe(df, use_container_width=True)
        
        # Download tombol grafik dan data CSV
        st.markdown("---")
        st.subheader("💾 Download Hasil")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="📊 Download Grafik Distribusi",
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
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(52, 73, 94, 0.1); border-radius: 15px; margin: 2rem 0;">
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
                    </ul>
                </div>
                <div style="text-align: center;">
                    <h4>🔬 Teknologi</h4>
                    <ul style="text-align: left; display: inline-block;">
                        <li>K-Means Clustering</li>
                        <li>OpenCV (cv2)</li>
                        <li>PIL (Python Imaging)</li>
                        <li>Matplotlib</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer kecil di bagian bawah halaman
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p>🎨 <strong>Dominant Color Picker</strong> - Dibuat dengan ❤️ oleh Clarisya Adeline</p>
        <p><em>Upload gambar dan temukan warna-warna dominan yang tersembunyi!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
