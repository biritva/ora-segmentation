import os
import io
import zipfile
import hashlib
import shutil
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path
from PIL import Image, ImageEnhance, UnidentifiedImageError
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
from huggingface_hub import hf_hub_download

# ==========================================
# УВЕЛИЧЕНИЕ ЛИМИТА PIL ДЛЯ БОЛЬШИХ ИЗОБРАЖЕНИЙ
# ==========================================
Image.MAX_IMAGE_PIXELS = 1_000_000_000

# Оптимизация памяти CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import pretrained_microscopy_models as pmm

# ==========================================
# КОНФИГУРАЦИЯ HUGGING FACE
# ==========================================
# ⚠️ ЗАМЕНИТЕ НА ВАШИ ДАННЫЕ:
HF_REPO_ID = "fomin33312/UnetPlusPlus_resnet50_high_lr"  # ← Ваш репозиторий
HF_FILENAME = "UnetPlusPlus_resnet50_high_lr.pth.tar"
HF_EXPECTED_SHA256 = ""  # ← SHA256 хеш (опционально)

# Токен НЕ нужен для публичных репозиториев
HF_TOKEN = None

# ==========================================
# ФИКСИРОВАННЫЕ ПАРАМЕТРЫ
# ==========================================
PATCH_SIZE = 2048
MAX_WIDTH = 1600
MODEL_STRIDE = 32
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_FILES = 50

# Параметры для ускорения PDF
PDF_IMAGE_MAX_WIDTH = 1200
PDF_JPEG_QUALITY = 85
PDF_MATPLOTLIB_DPI = 100

# Пути к модели
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / HF_FILENAME

# Параметры для ускорения PDF
PDF_IMAGE_MAX_WIDTH = 1200
PDF_JPEG_QUALITY = 85
PDF_MATPLOTLIB_DPI = 100

# Пути к модели
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / HF_FILENAME

# ==========================================
# КОНФИГУРАЦИЯ КЛАССОВ
# ==========================================
ACTIVE_CLASSES = [0, 1, 3, 5, 7]
NUM_ACTIVE_CLASSES = len(ACTIVE_CLASSES)

CLASS_NAMES = {
    0: "Силикаты",
    1: "Тальк",
    3: "Магнетит",
    5: "Рудная вкрапленность",
    7: "Другое"
}

ACTIVE_CLASS_COLORS = np.array([
    [0, 0, 0],        # Силикаты - чёрный (фон)
    [255, 0, 0],      # Тальк - красный
    [0, 255, 0],      # Магнетит - зелёный
    [0, 0, 255],      # Рудная вкрапленность - синий
    [255, 255, 0],    # Другое - жёлтый
], dtype=np.uint8)

st.set_page_config(
    page_title="Mosel Segmentation App",
    layout="wide",
    page_icon="🔬"
)

# ==========================================
# 📥 ФУНКЦИИ ЗАГРУЗКИ МОДЕЛИ С HUGGING FACE
# ==========================================

def compute_sha256(file_path):
    """Вычисляет SHA256 хеш файла для проверки целостности"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):  # 8 MB chunks
            sha256.update(chunk)
    return sha256.hexdigest()


def download_model_from_hf():
    """
    Загружает модель с Hugging Face, если её нет локально или она повреждена.
    Возвращает путь к модели.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Проверка: файл уже существует?
    if MODEL_PATH.exists():
        file_size_gb = MODEL_PATH.stat().st_size / (1024**3)
        
        if file_size_gb < 0.1:
            st.warning(f"⚠️ Файл слишком маленький ({file_size_gb:.2f} GB), удаляем...")
            MODEL_PATH.unlink()
        else:
            # Проверка целостности через SHA256 (если задан)
            if HF_EXPECTED_SHA256:
                st.info("🔍 Проверка целостности файла...")
                actual_hash = compute_sha256(MODEL_PATH)
                if actual_hash == HF_EXPECTED_SHA256:
                    st.success(f"✅ Модель уже загружена и проверена: {MODEL_PATH}")
                    return str(MODEL_PATH)
                else:
                    st.warning(f"⚠️ Хеш не совпадает! Файл повреждён.")
                    st.warning(f"   Ожидалось: {HF_EXPECTED_SHA256}")
                    st.warning(f"   Получено:  {actual_hash}")
                    MODEL_PATH.unlink()
            else:
                st.success(f"✅ Модель уже загружена: {MODEL_PATH} ({file_size_gb:.2f} GB)")
                return str(MODEL_PATH)
    
    # Загрузка с Hugging Face
    st.info(f"📥 Загрузка модели с Hugging Face...")
    st.info(f"   Репозиторий: `{HF_REPO_ID}`")
    st.info(f"   Файл: `{HF_FILENAME}`")
    st.info(f"   Это может занять 5-15 минут в зависимости от скорости интернета...")
    
    try:
        # Скачиваем в кэш HF
        cached_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="model",
            token=HF_TOKEN,
            cache_dir=".hf_cache",
            force_download=False,
        )
        
        # Копируем из кэша в models/
        st.info("📋 Копирование модели в локальную папку...")
        shutil.copy2(cached_path, MODEL_PATH)
        
        file_size_gb = MODEL_PATH.stat().st_size / (1024**3)
        st.success(f"✅ Модель успешно загружена: {MODEL_PATH}")
        st.success(f"📏 Размер: {file_size_gb:.2f} GB")
        
        # Проверка хеша после загрузки (если задан)
        if HF_EXPECTED_SHA256:
            actual_hash = compute_sha256(MODEL_PATH)
            if actual_hash == HF_EXPECTED_SHA256:
                st.success("✅ Целостность файла подтверждена")
            else:
                st.error("❌ Хеш не совпадает! Файл может быть повреждён.")
        
        return str(MODEL_PATH)
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели с Hugging Face: {e}")
        st.info("""
        **Возможные решения:**
        1. Проверьте интернет-соединение
        2. Убедитесь, что репозиторий публичный (или укажите токен)
        3. Проверьте правильность `HF_REPO_ID` и `HF_FILENAME`
        4. Загрузите модель вручную в папку `models/`
        """)
        raise


# ==========================================
# 🔤 ПОДКЛЮЧЕНИЕ ШРИФТА С КИРИЛЛИЦЕЙ
# ==========================================

def find_cyrillic_font():
    """Ищет шрифт с поддержкой кириллицы в системе"""
    font_candidates = [
        "fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    
    local_fonts = list(Path('.').glob('fonts/*.ttf')) + list(Path('.').glob('*.ttf'))
    font_candidates.extend([str(f) for f in local_fonts])
    
    for font_path in font_candidates:
        if os.path.exists(font_path):
            return font_path
    return None


def register_cyrillic_font():
    """Регистрирует шрифт с поддержкой кириллицы в ReportLab"""
    font_path = find_cyrillic_font()
    
    if font_path is None:
        return "Helvetica"
    
    try:
        font_name = "CyrillicFont"
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        return font_name
    except Exception as e:
        print(f"❌ Ошибка регистрации шрифта: {e}")
        return "Helvetica"


CYRILLIC_FONT = register_cyrillic_font()

# ==========================================
# 1. УТИЛИТЫ ДЛЯ РАБОТЫ С РАЗМЕРАМИ
# ==========================================

def round_up_to_multiple(value, multiple=32):
    """Округляет значение вверх до кратного multiple"""
    return ((value + multiple - 1) // multiple) * multiple


def pad_to_multiple(image, multiple=32, pad_value=0):
    """Добавляет padding справа и снизу, чтобы размеры стали кратны multiple"""
    h, w = image.shape[:2]
    new_h = round_up_to_multiple(h, multiple)
    new_w = round_up_to_multiple(w, multiple)

    if new_h == h and new_w == w:
        return image, (h, w)

    if len(image.shape) == 3:
        padded = np.full((new_h, new_w, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        padded = np.full((new_h, new_w), pad_value, dtype=image.dtype)

    padded[:h, :w] = image
    return padded, (h, w)


def crop_to_original(image, crop_info):
    """Обрезает padding, возвращая исходный размер"""
    orig_h, orig_w = crop_info
    return image[:orig_h, :orig_w]


# ==========================================
# ⚡ УТИЛИТЫ ДЛЯ УСКОРЕНИЯ PDF
# ==========================================

def resize_for_pdf(image, max_width=PDF_IMAGE_MAX_WIDTH):
    """Уменьшает изображение до max_width с сохранением пропорций"""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def image_to_jpeg_buffer(image, quality=PDF_JPEG_QUALITY):
    """Конвертирует numpy array в JPEG буфер"""
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    pil_img = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)
    return buffer


# ==========================================
# 2. ФУНКЦИИ ЗАГРУЗКИ И ПРЕДОБРАБОТКИ
# ==========================================

@st.cache_resource
def load_segmentation_model():
    """Загрузка модели с автозагрузкой с Hugging Face"""
    
    # Проверяем наличие модели, если нет — загружаем с HF
    if not MODEL_PATH.exists():
        st.warning("⚠️ Модель не найдена локально. Начинается загрузка с Hugging Face (~2 ГБ)...")
        try:
            download_model_from_hf()
        except Exception as e:
            st.error(f"❌ Не удалось загрузить модель: {e}")
            st.stop()
    
    # Загрузка модели в память
    with st.spinner("🧠 Загрузка модели в память..."):
        model, preprocessing_fn = pmm.segmentation_training.load_segmentation_model(
            MODEL_PATH, classes=12
        )
    
    return model, preprocessing_fn


def preprocess_image(img_array, enhance_contrast, enhance_brightness, enhance_color, scale_channels):
    """Предобработка изображения"""
    pil_img = Image.fromarray(img_array)

    if enhance_contrast:
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.5)
    if enhance_brightness:
        pil_img = ImageEnhance.Brightness(pil_img).enhance(1.2)
    if enhance_color:
        pil_img = ImageEnhance.Color(pil_img).enhance(1.3)

    im_result = np.array(pil_img).astype(np.float32)

    if scale_channels:
        im_result[:, :, 0] *= 1.10  # R +10%
        im_result[:, :, 1] *= 0.95  # G -5%
        im_result[:, :, 2] *= 1.10  # B +10%

    return np.clip(im_result, 0, 255).astype(np.uint8)


def resize_to_max_width(image, max_width):
    """Уменьшает изображение до фиксированной ширины с сохранением пропорций"""
    h, w = image.shape[:2]
    original_size = (w, h)
    scaled = False

    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        new_h = round_up_to_multiple(new_h, MODEL_STRIDE)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scaled = True
    else:
        new_h = round_up_to_multiple(h, MODEL_STRIDE)
        if new_h != h:
            image = cv2.resize(image, (w, new_h), interpolation=cv2.INTER_AREA)
            scaled = True

    new_size = (image.shape[1], image.shape[0])
    aspect_ratio = original_size[0] / original_size[1] if original_size[1] > 0 else 1.0

    return image, {
        'original_size': original_size,
        'new_size': new_size,
        'scaled': scaled,
        'aspect_ratio': aspect_ratio,
    }


def filter_predictions_by_classes(pred, active_classes):
    """Фильтрует предсказания модели, оставляя только нужные классы"""
    return pred[..., active_classes]


def get_class_map(filtered_pred):
    """Получает карту классов из отфильтрованных предсказаний"""
    class_map = np.argmax(filtered_pred, axis=-1)

    actual_class_map = np.zeros_like(class_map, dtype=np.int32)
    for local_idx, global_idx in enumerate(ACTIVE_CLASSES):
        actual_class_map[class_map == local_idx] = global_idx

    color_overlay = ACTIVE_CLASS_COLORS[class_map]

    return class_map, actual_class_map, color_overlay


# ==========================================
# 3. ГЕНЕРАЦИЯ PDF ОТЧЁТА
# ==========================================

def generate_pdf_report(original_im, class_map, actual_class_map, color_overlay, df_stats, image_info, filename=""):
    """Генерация PDF отчёта с поддержкой кириллицы и оптимизацией скорости"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontName=CYRILLIC_FONT, fontSize=20,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12, alignment=TA_CENTER, leading=24
    )

    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'],
        fontName=CYRILLIC_FONT, fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10, spaceBefore=10, leading=18
    )

    cyrillic_normal = ParagraphStyle(
        'CyrillicNormal', parent=styles['Normal'],
        fontName=CYRILLIC_FONT, fontSize=10, leading=14
    )

    cyrillic_cell = ParagraphStyle(
        'CyrillicCell', parent=styles['Normal'],
        fontName=CYRILLIC_FONT, fontSize=9,
        leading=12, alignment=TA_CENTER
    )

    # Уменьшение изображений для PDF
    original_for_pdf = resize_for_pdf(original_im, max_width=PDF_IMAGE_MAX_WIDTH)
    overlay_for_pdf = resize_for_pdf(color_overlay, max_width=PDF_IMAGE_MAX_WIDTH)
    
    pdf_h, pdf_w = original_for_pdf.shape[:2]
    pdf_aspect = pdf_w / pdf_h
    
    if pdf_aspect > 1:
        img_width_pdf = 6*inch
        img_height_pdf = img_width_pdf / pdf_aspect
    else:
        img_height_pdf = 5*inch
        img_width_pdf = img_height_pdf * pdf_aspect

    # Титульная страница
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("Отчёт по сегментации микроскопических изображений", title_style))
    if filename:
        story.append(Paragraph(f"Файл: {filename}", cyrillic_normal))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Дата анализа: {datetime.now().strftime('%d.%m.%Y %H:%M')}", cyrillic_normal))

    orig_w, orig_h = image_info['original_size']
    new_w, new_h = image_info['new_size']
    story.append(Paragraph(f"Исходный размер: {orig_w} x {orig_h} пикселей", cyrillic_normal))
    story.append(Paragraph(f"Размер после обработки: {new_w} x {new_h} пикселей", cyrillic_normal))
    story.append(Paragraph(f"Соотношение сторон: {image_info['aspect_ratio']:.2f}", cyrillic_normal))
    story.append(Paragraph(
        f"Активные классы: {', '.join([CLASS_NAMES[c] for c in ACTIVE_CLASSES])}",
        cyrillic_normal
    ))
    story.append(Paragraph(f"Размер патча: {PATCH_SIZE}", cyrillic_normal))
    story.append(Spacer(1, 0.5*inch))

    # Оригинальное изображение
    story.append(Paragraph("1. Оригинальное изображение", heading_style))
    img_buffer = image_to_jpeg_buffer(original_for_pdf)
    story.append(RLImage(img_buffer, width=img_width_pdf, height=img_height_pdf))

    # Карта классов
    story.append(PageBreak())
    story.append(Paragraph("2. Карта классов (наложение масок)", heading_style))
    overlay_buffer = image_to_jpeg_buffer(overlay_for_pdf)
    story.append(RLImage(overlay_buffer, width=img_width_pdf, height=img_height_pdf))

    # Маски
    story.append(PageBreak())
    story.append(Paragraph("3. Отдельные маски для активных классов", heading_style))

    mask_max_width = 300
    fig, axes = plt.subplots(1, NUM_ACTIVE_CLASSES, figsize=(12, 3))
    fig.suptitle('Маски по классам', fontsize=14, fontweight='bold')

    for idx, (ax, class_id) in enumerate(zip(axes, ACTIVE_CLASSES)):
        mask_full = (actual_class_map == class_id).astype(np.uint8) * 255
        mask_small = resize_for_pdf(mask_full, max_width=mask_max_width)
        ax.imshow(mask_small, cmap='gray')
        ax.set_title(CLASS_NAMES[class_id], fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    masks_buffer = io.BytesIO()
    plt.savefig(masks_buffer, format='PNG', dpi=PDF_MATPLOTLIB_DPI, bbox_inches='tight')
    plt.close()
    masks_buffer.seek(0)
    story.append(RLImage(masks_buffer, width=6.5*inch, height=2*inch))

    # Статистика
    story.append(PageBreak())
    story.append(Paragraph("4. Статистика по активным классам", heading_style))
    story.append(Spacer(1, 0.1*inch))

    table_data = [
        [
            Paragraph('<b>Класс</b>', cyrillic_cell),
            Paragraph('<b>Название</b>', cyrillic_cell),
            Paragraph('<b>Пиксели</b>', cyrillic_cell),
            Paragraph('<b>Площадь (%)</b>', cyrillic_cell)
        ]
    ]
    
    for _, row in df_stats.iterrows():
        table_data.append([
            Paragraph(str(row['Класс']), cyrillic_cell),
            Paragraph(row['Название'], cyrillic_cell),
            Paragraph(f"{int(row['Пиксели']):,}", cyrillic_cell),
            Paragraph(f"{row['Площадь (%)']:.2f}%", cyrillic_cell)
        ])

    table = Table(table_data, colWidths=[0.8*inch, 2.2*inch, 1.5*inch, 1.2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    story.append(table)
    story.append(Spacer(1, 0.3*inch))

    # График
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    
    fig_plot, ax = plt.subplots(figsize=(8, 4))
    df_plot = df_stats[df_stats['Класс'] != 0].copy()
    if not df_plot.empty:
        labels = [CLASS_NAMES[c] for c in df_plot['Класс']]
        ax.bar(labels, df_plot['Площадь (%)'], color='#1f77b4')
        ax.set_xlabel('Класс', fontsize=11)
        ax.set_ylabel('Площадь (%)', fontsize=11)
        ax.set_title('Распределение площадей по активным классам (без фона)',
                     fontsize=13, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='PNG', dpi=PDF_MATPLOTLIB_DPI, bbox_inches='tight')
        plt.close()
        plot_buffer.seek(0)
        story.append(RLImage(plot_buffer, width=6*inch, height=3*inch))

    story.append(Spacer(1, 0.3*inch))
    total_pixels = actual_class_map.size
    story.append(Paragraph(f"<b>Общее количество пикселей:</b> {total_pixels:,}", cyrillic_normal))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================
# 🖼️ ФУНКЦИЯ ОТОБРАЖЕНИЯ РЕЗУЛЬТАТОВ ДЛЯ ОДНОГО ИЗОБРАЖЕНИЯ
# ==========================================

def display_single_result(result, idx):
    """Отображает детальные результаты для одного изображения"""
    filename = result['filename']
    original_im = result['original_im']
    class_map = result['class_map']
    actual_class_map = result['actual_class_map']
    color_overlay = result['color_overlay']
    image_info = result['image_info']
    df_stats = result['df_stats']
    
    orig_w, orig_h = image_info['original_size']
    res_w, res_h = class_map.shape[1], class_map.shape[0]
    
    # Вычисляем доминирующий класс (кроме фона)
    df_no_bg = df_stats[df_stats['Класс'] != 0]
    if not df_no_bg.empty:
        dominant = df_no_bg.loc[df_no_bg['Площадь (%)'].idxmax()]
        dominant_info = f"{dominant['Название']} ({dominant['Площадь (%)']:.1f}%)"
    else:
        dominant_info = "—"
    
    expander_title = f"📷 [{idx+1}] {filename}  |  {orig_w}×{orig_h}  |  Доминирующий: {dominant_info}"
    
    with st.expander(expander_title, expanded=(idx == 0)):
        # Информация о файле
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Исходный размер", f"{orig_w} × {orig_h}")
        with info_col2:
            st.metric("Размер результата", f"{res_w} × {res_h}")
        with info_col3:
            st.metric("Соотношение сторон", f"{image_info['aspect_ratio']:.2f}")
        with info_col4:
            total_pixels = actual_class_map.size
            st.metric("Всего пикселей", f"{total_pixels:,}")
        
        st.divider()
        
        # Изображения
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("**📷 Оригинальное изображение**")
            st.image(original_im, use_container_width=True, caption=f"{orig_w}×{orig_h}")
        with img_col2:
            st.markdown("**🎨 Карта классов**")
            st.image(color_overlay, use_container_width=True, caption=f"{res_w}×{res_h}")
        
        # Легенда классов
        st.markdown("**🎨 Легенда классов:**")
        legend_cols = st.columns(NUM_ACTIVE_CLASSES)
        for col_idx, (col, class_id) in enumerate(zip(legend_cols, ACTIVE_CLASSES)):
            color_hex = '#{:02x}{:02x}{:02x}'.format(*ACTIVE_CLASS_COLORS[col_idx])
            col.markdown(f"""
            <div style='background-color: {color_hex}; padding: 8px; border-radius: 5px; text-align: center; color: white; font-size: 12px;'>
                <b>{CLASS_NAMES[class_id]}</b>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Статистика и график
        stat_col1, stat_col2 = st.columns([1, 1])
        
        with stat_col1:
            st.markdown("**📊 Таблица статистики**")
            st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        with stat_col2:
            st.markdown("**📈 Распределение площадей**")
            df_plot = df_stats[df_stats['Класс'] != 0].copy()
            if not df_plot.empty:
                fig, ax = plt.subplots(figsize=(6, 3))
                labels = [CLASS_NAMES[c] for c in df_plot['Класс']]
                bar_colors = [ACTIVE_CLASS_COLORS[ACTIVE_CLASSES.index(c)] / 255 for c in df_plot['Класс']]
                ax.bar(labels, df_plot['Площадь (%)'], color=bar_colors)
                ax.set_ylabel('Площадь (%)', fontsize=10)
                ax.set_title(f'Распределение (без фона)', fontsize=11)
                plt.xticks(rotation=30, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Нет ненулевых классов (только фон)")
        
        st.divider()
        
        # Кнопки скачивания
        st.markdown("**💾 Скачать результаты для этого файла:**")
        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
        
        filename_stem = Path(filename).stem
        
        with dl_col1:
            csv = df_stats.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 CSV статистика",
                data=csv,
                file_name=f"{filename_stem}_stats.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"csv_{idx}"
            )
        
        with dl_col2:
            img_pil = Image.fromarray(color_overlay.astype(np.uint8))
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            st.download_button(
                label="🖼️ PNG карта",
                data=buf.getvalue(),
                file_name=f"{filename_stem}_overlay.png",
                mime="image/png",
                use_container_width=True,
                key=f"png_{idx}"
            )
        
        with dl_col3:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for class_id in ACTIVE_CLASSES:
                    mask = (actual_class_map == class_id).astype(np.uint8) * 255
                    img_byte_arr = io.BytesIO()
                    Image.fromarray(mask).save(img_byte_arr, format='PNG')
                    safe_name = CLASS_NAMES[class_id].replace(' ', '_')
                    zip_file.writestr(f"mask_{safe_name}.png", img_byte_arr.getvalue())
            st.download_button(
                label="📦 Маски (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{filename_stem}_masks.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"zip_{idx}"
            )
        
        with dl_col4:
            with st.spinner("Генерация PDF..."):
                pdf_data = generate_pdf_report(
                    original_im, class_map, actual_class_map, color_overlay,
                    df_stats, image_info, filename=filename
                )
            st.download_button(
                label="📑 PDF отчёт",
                data=pdf_data,
                file_name=f"{filename_stem}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"pdf_{idx}"
            )


# ==========================================
# 4. ОСНОВНОЙ ИНТЕРФЕЙС STREAMLIT
# ==========================================

st.title("🔬 Mosel: Сегментация микроскопических изображений")
st.markdown(f"""
Загрузите **одно или несколько** изображений для запуска модели сегментации.  
**Активные классы:** {', '.join([CLASS_NAMES[c] for c in ACTIVE_CLASSES])}  
**Параметры (фиксированные):** `patch_size = {PATCH_SIZE}` | `max_width = {MAX_WIDTH}px`
""")

# Информация о шрифте и модели
col_info1, col_info2 = st.columns(2)
with col_info1:
    if CYRILLIC_FONT != "Helvetica":
        st.success(f"✅ Шрифт для PDF: `{CYRILLIC_FONT}`")
    else:
        st.error("❌ Шрифт с кириллицей не найден!")
with col_info2:
    if MODEL_PATH.exists():
        size_gb = MODEL_PATH.stat().st_size / (1024**3)
        st.success(f"✅ Модель загружена ({size_gb:.2f} GB)")
    else:
        st.info(f"📥 Модель будет загружена с Hugging Face при первом запуске")

with st.sidebar:
    st.header("⚙️ Настройки")

    # Информация о модели
    st.subheader("🧠 Модель")
    if MODEL_PATH.exists():
        size_gb = MODEL_PATH.stat().st_size / (1024**3)
        st.success(f"✅ Модель загружена")
        st.caption(f"📏 Размер: {size_gb:.2f} GB")
        st.caption(f"📍 Источник: Hugging Face")
        st.caption(f"🔗 Репозиторий: `{HF_REPO_ID}`")
    else:
        st.warning("⚠️ Модель будет загружена при первом запуске")
        st.caption(f"🔗 Репозиторий: `{HF_REPO_ID}`")
    
    st.divider()

    st.subheader("📏 Фиксированные параметры")
    st.info(f"""
    - **Размер патча:** `{PATCH_SIZE}px`
    - **Макс. ширина:** `{MAX_WIDTH}px`
    - **Макс. размер файла:** `{MAX_FILE_SIZE_MB} MB`
    - **Макс. файлов:** `{MAX_FILES}`
    - **Stride модели:** `{MODEL_STRIDE}`
    """)

    st.subheader("🎯 Активные классы")
    for idx, class_id in enumerate(ACTIVE_CLASSES):
        color_hex = '#{:02x}{:02x}{:02x}'.format(*ACTIVE_CLASS_COLORS[idx])
        st.markdown(f"""
        <div style='display: flex; align-items: center; margin: 5px 0;'>
            <div style='width: 20px; height: 20px; background-color: {color_hex}; margin-right: 10px; border-radius: 3px; border: 1px solid #ccc;'></div>
            <span><b>Class {class_id}:</b> {CLASS_NAMES[class_id]}</span>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Предобработка изображения")
    enhance_contrast = st.checkbox("Увеличить контраст (x1.5)", value=False)
    enhance_brightness = st.checkbox("Увеличить яркость (x1.2)", value=False)
    enhance_color = st.checkbox("Увеличить цветность (x1.3)", value=False)
    scale_channels = st.checkbox("Коррекция каналов (R+10%, G-5%, B+10%)", value=False)

    st.divider()
    st.info(f"Модель: `UnetPlusPlus_resnet50_high_lr.pth.tar`")
    st.warning(f"⚠️ **Большой патч ({PATCH_SIZE}px)** — требуется 12-16 GB VRAM")

    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
        st.info(f"Свободно: {free_mem / 1024**3:.2f} / {total_mem / 1024**3:.2f} GB")
        if free_mem < 8 * 1024**3:
            st.error("⚠️ Мало свободной памяти для патча 2048!")
    else:
        st.warning("Используется CPU (будет очень медленно)")

# Загрузка модели
model, preprocessing_fn = load_segmentation_model()

# ==========================================
# 📦 ЗАГРУЗКА НЕСКОЛЬКИХ ФАЙЛОВ
# ==========================================

uploaded_files = st.file_uploader(
    "Выберите одно или несколько изображений...",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True,
    help=f"Можно загрузить до {MAX_FILES} файлов. Максимальный размер каждого: {MAX_FILE_SIZE_MB} МБ"
)

if uploaded_files:
    st.success(f"✅ Загружено файлов: **{len(uploaded_files)}**")
    
    if len(uploaded_files) > MAX_FILES:
        st.error(f"⛔ Слишком много файлов! Максимум: {MAX_FILES}")
        st.stop()
    
    # Проверка размера каждого файла
    valid_files = []
    for uploaded_file in uploaded_files:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error(f"❌ Файл `{uploaded_file.name}` слишком большой: {file_size_mb:.2f} МБ")
        else:
            valid_files.append(uploaded_file)
    
    if not valid_files:
        st.stop()
    
    uploaded_files = valid_files
    
    # ==========================================
    # 🚀 ПАКЕТНАЯ ОБРАБОТКА
    # ==========================================
    
    if st.button("🚀 Запустить сегментацию всех изображений", type="primary", use_container_width=True):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        failed_files = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"🔄 Обработка {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            try:
                image = Image.open(uploaded_file).convert("RGB")
                total_pixels = image.width * image.height
                
                if total_pixels > Image.MAX_IMAGE_PIXELS:
                    failed_files.append({
                        'name': uploaded_file.name,
                        'error': f"Слишком большое: {total_pixels:,} пикселей"
                    })
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    continue
                
                original_im = np.array(image)
                del image
                
                processed_im = preprocess_image(
                    original_im,
                    enhance_contrast, enhance_brightness, enhance_color, scale_channels
                )
                
                processed_im, image_info = resize_to_max_width(processed_im, max_width=MAX_WIDTH)
                processed_im, crop_info = pad_to_multiple(processed_im, multiple=MODEL_STRIDE, pad_value=0)
                
                pred_full = pmm.segmentation_training.segmentation_models_inference(
                    processed_im, model, preprocessing_fn,
                    batch_size=1, patch_size=PATCH_SIZE, num_classes=12, probabilities=None
                )
                
                pred_filtered = filter_predictions_by_classes(pred_full, ACTIVE_CLASSES)
                class_map_padded, actual_class_map_padded, color_overlay_padded = get_class_map(pred_filtered)
                
                class_map = crop_to_original(class_map_padded, crop_info)
                actual_class_map = crop_to_original(actual_class_map_padded, crop_info)
                color_overlay = crop_to_original(color_overlay_padded, crop_info)
                
                total_pixels_seg = actual_class_map.size
                stats_data = []
                for class_id in ACTIVE_CLASSES:
                    pixel_count = np.sum(actual_class_map == class_id)
                    percentage = (pixel_count / total_pixels_seg) * 100
                    stats_data.append({
                        'Класс': class_id,
                        'Название': CLASS_NAMES[class_id],
                        'Пиксели': int(pixel_count),
                        'Площадь (%)': round(percentage, 2)
                    })
                
                df_stats = pd.DataFrame(stats_data)
                
                all_results.append({
                    'filename': uploaded_file.name,
                    'original_im': original_im,
                    'class_map': class_map,
                    'actual_class_map': actual_class_map,
                    'color_overlay': color_overlay,
                    'image_info': image_info,
                    'df_stats': df_stats,
                    'crop_info': crop_info
                })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
            except torch.cuda.OutOfMemoryError as e:
                failed_files.append({
                    'name': uploaded_file.name,
                    'error': 'CUDA Out of Memory'
                })
                torch.cuda.empty_cache()
                progress_bar.progress((idx + 1) / len(uploaded_files))
            except Exception as e:
                failed_files.append({
                    'name': uploaded_file.name,
                    'error': str(e)
                })
                progress_bar.progress((idx + 1) / len(uploaded_files))
                continue
        
        status_text.text("✅ Обработка завершена!")
        
        # Сохраняем результаты в session_state
        st.session_state['all_results'] = all_results
        st.session_state['failed_files'] = failed_files
        st.session_state['total_files'] = len(uploaded_files)
    
    # ==========================================
    # 📊 ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    # ==========================================
    
    if 'all_results' in st.session_state and st.session_state['all_results']:
        all_results = st.session_state['all_results']
        failed_files = st.session_state.get('failed_files', [])
        total_files = st.session_state.get('total_files', len(uploaded_files))
        
        st.divider()
        st.header(f"📊 Результаты обработки ({len(all_results)}/{total_files})")
        
        # Отчёт об ошибках
        if failed_files:
            with st.expander(f"❌ Ошибки обработки ({len(failed_files)})", expanded=False):
                for fail in failed_files:
                    st.error(f"- `{fail['name']}`: {fail['error']}")
        
        # Сводная статистика
        st.subheader("📈 Сводная таблица по всем изображениям")
        
        summary_data = []
        for result in all_results:
            df_stats = result['df_stats']
            df_no_bg = df_stats[df_stats['Класс'] != 0]
            
            if not df_no_bg.empty:
                dominant = df_no_bg.loc[df_no_bg['Площадь (%)'].idxmax()]
                dominant_name = dominant['Название']
                dominant_pct = f"{dominant['Площадь (%)']:.2f}%"
            else:
                dominant_name = "—"
                dominant_pct = "—"
            
            orig_w, orig_h = result['image_info']['original_size']
            
            summary_data.append({
                '№': len(summary_data) + 1,
                'Файл': result['filename'],
                'Размер': f"{orig_w}×{orig_h}",
                'Доминирующий класс': dominant_name,
                'Доля (%)': dominant_pct,
                'Тальк (%)': f"{df_stats[df_stats['Класс']==1]['Площадь (%)'].values[0]:.2f}",
                'Магнетит (%)': f"{df_stats[df_stats['Класс']==3]['Площадь (%)'].values[0]:.2f}",
                'Рудная вр. (%)': f"{df_stats[df_stats['Класс']==5]['Площадь (%)'].values[0]:.2f}",
                'Другое (%)': f"{df_stats[df_stats['Класс']==7]['Площадь (%)'].values[0]:.2f}",
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        # Кнопки пакетного скачивания
        st.subheader("📥 Пакетное скачивание")
        batch_col1, batch_col2, batch_col3 = st.columns(3)
        
        with batch_col1:
            csv_summary = df_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Сводная таблица (CSV)",
                data=csv_summary,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with batch_col2:
            if st.button("📦 Собрать ZIP со всеми результатами", use_container_width=True):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for result in all_results:
                        filename_base = Path(result['filename']).stem
                        
                        img_pil = Image.fromarray(result['color_overlay'].astype(np.uint8))
                        buf = io.BytesIO()
                        img_pil.save(buf, format="PNG")
                        zip_file.writestr(f"{filename_base}_overlay.png", buf.getvalue())
                        
                        csv = result['df_stats'].to_csv(index=False).encode('utf-8')
                        zip_file.writestr(f"{filename_base}_stats.csv", csv)
                        
                        for class_id in ACTIVE_CLASSES:
                            mask = (result['actual_class_map'] == class_id).astype(np.uint8) * 255
                            img_byte_arr = io.BytesIO()
                            Image.fromarray(mask).save(img_byte_arr, format='PNG')
                            safe_name = CLASS_NAMES[class_id].replace(' ', '_')
                            zip_file.writestr(f"{filename_base}_mask_{safe_name}.png", img_byte_arr.getvalue())
                
                st.download_button(
                    label="💾 Скачать ZIP архив",
                    data=zip_buffer.getvalue(),
                    file_name=f"segmentation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        
        with batch_col3:
            if st.button("📑 Собрать ZIP с PDF отчётами", use_container_width=True):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for result in all_results:
                        pdf_data = generate_pdf_report(
                            result['original_im'],
                            result['class_map'],
                            result['actual_class_map'],
                            result['color_overlay'],
                            result['df_stats'],
                            result['image_info'],
                            filename=result['filename']
                        )
                        zip_file.writestr(f"{Path(result['filename']).stem}_report.pdf", pdf_data)
                
                st.download_button(
                    label="💾 Скачать PDF архив",
                    data=zip_buffer.getvalue(),
                    file_name=f"segmentation_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        
        # Детальное отображение для каждого изображения
        st.divider()
        st.header(f"🖼️ Детальные результаты по каждому изображению ({len(all_results)})")
        st.info("💡 Нажмите на заголовок изображения, чтобы развернуть/свернуть детальный просмотр")
        
        for idx, result in enumerate(all_results):
            display_single_result(result, idx)

else:
    st.info("👆 Пожалуйста, загрузите одно или несколько изображений для начала работы.")
    
    st.markdown(f"""
    ### 📋 Информация о конфигурации
    
    | Параметр | Значение |
    |----------|----------|
    | Активные классы | {', '.join([f"{c} ({CLASS_NAMES[c]})" for c in ACTIVE_CLASSES])} |
    | **Размер патча** | **{PATCH_SIZE}** (фиксирован) |
    | **Макс. ширина** | **{MAX_WIDTH}** (фиксирована) |
    | **Макс. размер файла** | **{MAX_FILE_SIZE_MB} МБ** |
    | **Макс. количество файлов** | **{MAX_FILES}** |
    | **Источник модели** | **Hugging Face: `{HF_REPO_ID}`** |
    | Модель | UnetPlusPlus + ResNet50 |
    | Stride | {MODEL_STRIDE} |
    | Поддержка aspect ratios | ✅ Любые |
    | Авто-padding | ✅ До кратных {MODEL_STRIDE} |
    
    ### 🎯 Классы сегментации
    
    | Класс | Название | Цвет |
    |-------|----------|------|
    | 0 | Силикаты | ⬛ Чёрный |
    | 1 | Тальк | 🟥 Красный |
    | 3 | Магнетит | 🟩 Зелёный |
    | 5 | Рудная вкрапленность | 🟦 Синий |
    | 7 | Другое | 🟨 Жёлтый |
    
    ### 📥 Автозагрузка модели
    
    При первом запуске приложение **автоматически загрузит модель** (~2 ГБ) с Hugging Face:
    - Репозиторий: `{HF_REPO_ID}`
    - Файл: `{HF_FILENAME}`
    - Загрузка занимает 5-15 минут
    - Модель сохраняется локально для будущих запусков
    """)
