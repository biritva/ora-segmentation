# app.py
import streamlit as st
import os
import sys
import subprocess
import json
import glob
import tempfile
import shutil
import atexit
from datetime import datetime
from pathlib import Path

# ============================================
# ГЛОБАЛЬНЫЕ ИМПОРТЫ (ОБЯЗАТЕЛЬНО!)
# ============================================

import torch
import clip
import numpy as np
import pandas as pd
import cv2
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from skimage import exposure
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from matplotlib.patches import Patch

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Настройка страницы
st.set_page_config(
    page_title="Анализ руд",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Создаем временные директории
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'results_dir' not in st.session_state:
    st.session_state.results_dir = os.path.join(st.session_state.temp_dir, 'results')
    os.makedirs(st.session_state.results_dir, exist_ok=True)
if 'pdf_dir' not in st.session_state:
    st.session_state.pdf_dir = os.path.join(st.session_state.results_dir, 'pdf_reports')
    os.makedirs(st.session_state.pdf_dir, exist_ok=True)

# Инициализация session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = {}
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# ============================================
# ФУНКЦИИ ПРОВЕРКИ БИБЛИОТЕК
# ============================================

def check_libraries():
    """Проверка установленных библиотек"""
    libraries = {
        'streamlit': 'Streamlit',
        'torch': 'PyTorch',
        'segment_anything': 'Segment Anything',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'reportlab': 'ReportLab',
        'tqdm': 'Tqdm',
        'skimage': 'Scikit-Image',
    }
    
    results = {}
    for lib, name in libraries.items():
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'unknown')
            results[name] = {'status': '✅', 'version': version}
        except ImportError:
            results[name] = {'status': '❌', 'version': 'не установлена'}
        except Exception as e:
            results[name] = {'status': '️', 'version': f'ошибка: {str(e)[:50]}'}
    
    return results

# ============================================
# ФУНКЦИИ ЗАГРУЗКИ МОДЕЛЕЙ
# ============================================

@st.cache_resource
def load_models():
    """Загрузка моделей SAM и CLIP"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Загрузка SAM
    with st.spinner('📥 Загрузка SAM-vit-b...'):
        sam_checkpoint = 'sam_vit_b_01ec64.pth'
        if not os.path.exists(sam_checkpoint):
            st.info('Скачивание весов SAM (~375 МБ)...')
            import urllib.request
            urllib.request.urlretrieve(
                'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
                sam_checkpoint
            )
        
        sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
        sam.to(device)
        sam.eval()
        
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.90,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200,
        )
    
    # Загрузка CLIP
    with st.spinner(' Загрузка CLIP ViT-B/32...'):
        clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
        clip_model.eval()
    
    return mask_generator, clip_model, clip_preprocess, device

# ============================================
# ФУНКЦИИ ОБРАБОТКИ ИЗОБРАЖЕНИЙ
# ============================================

def preprocess_image(image_path, config):
    """Предобработка изображения"""
    img_pil = PILImage.open(image_path)
    
    if img_pil.mode != 'RGB':
        if img_pil.mode == 'I;16':
            arr = np.array(img_pil).astype(np.float32)
            arr = (arr / arr.max() * 255).astype(np.uint8)
            img_pil = PILImage.fromarray(arr).convert('RGB')
        else:
            img_pil = img_pil.convert('RGB')
    
    original_np = np.array(img_pil)
    original_size = original_np.shape[:2]
    
    max_size = config.get('max_image_size', 2048)
    scale_factor = 1.0
    if max_size > 0:
        h, w = original_np.shape[:2]
        if max(h, w) > max_size:
            scale_factor = max_size / max(h, w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            img_pil = img_pil.resize((new_w, new_h), PILImage.LANCZOS)
            original_np = np.array(img_pil)
    
    work = original_np.copy()
    
    if config.get('clahe_enabled', True):
        lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        work = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    if config.get('denoise_enabled', True):
        work = cv2.fastNlMeansDenoisingColored(work, None, 5, 5, 7, 21)
    
    return original_np, work, {'scale_factor': scale_factor, 'original_size': original_size}

def classify_segment(image_np, mask, clip_model, clip_preprocess, class_names, class_descriptions, device):
    """Классификация сегмента"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, 0.0, {}
    
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    h, w = image_np.shape[:2]
    
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    patch = image_np[y1:y2, x1:x2].copy()
    patch_mask = mask[y1:y2, x1:x2]
    patch[~patch_mask] = [128, 128, 128]
    
    patch_pil = PILImage.fromarray(patch)
    patch_input = clip_preprocess(patch_pil).unsqueeze(0).to(device)
    
    text_inputs = torch.cat([
        clip.tokenize(f'a photo of {desc}') for desc in class_descriptions
    ]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(patch_input)
        text_features = clip_model.encode_text(text_inputs)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values = similarity[0].cpu().numpy()
    
    best_idx = np.argmax(values)
    return class_names[best_idx], float(values[best_idx]), {class_names[i]: float(values[i]) for i in range(len(class_names))}

def process_image(image_file, mask_generator, clip_model, clip_preprocess, device, config):
    """Обработка одного изображения"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(image_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        original_np, processed_np, preprocess_info = preprocess_image(tmp_path, config)
        
        masks_data = mask_generator.generate(processed_np)
        masks_data = [m for m in masks_data if m['area'] >= config.get('min_segment_area', 300)]
        
        CLASSES = {
            'sulfide_massive': 'bright metallic sulfide mineral pyrite chalcopyrite massive aggregate',
            'sulfide_disseminated': 'small scattered sulfide grains disseminated in rock',
            'magnetite': 'black opaque magnetite Fe3O4 iron oxide',
            'other_oxide': 'dark oxide mineral hematite ilmenite chromite',
            'talc': 'dark green gray talc Mg3Si4O10(OH)2 scattered',
            'silicate_matrix': 'light gray silicate rock matrix',
            'serpentinite': 'green serpentinite altered rock',
            'fracture': 'fracture crack void in rock',
            'void': 'empty hole cavity',
        }
        
        class_names = list(CLASSES.keys())
        class_descriptions = list(CLASSES.values())
        
        segments = []
        for mask_info in masks_data:
            mask = mask_info['segmentation']
            area_px = int(mask_info['area'])
            
            best_class, confidence, _ = classify_segment(
                processed_np, mask, clip_model, clip_preprocess,
                class_names, class_descriptions, device
            )
            
            if confidence < 0.20 or best_class == 'silicate_matrix':
                continue
            
            segments.append({
                'mask': mask,
                'class': best_class,
                'confidence': confidence,
                'area_px': area_px,
                'bbox': mask_info['bbox'],
            })
        
        return segments, original_np, preprocess_info
    
    finally:
        os.unlink(tmp_path)

# ============================================
# ГЕНЕРАЦИЯ PDF
# ============================================

def generate_pdf(segments, original_np, img_name, output_path):
    """Генерация PDF отчёта"""
    
    # Регистрация шрифта
    FONT_PATHS = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans.ttf',
        'C:\\Windows\\Fonts\\arial.ttf',
    ]
    
    font_path = None
    for path in FONT_PATHS:
        if os.path.exists(path):
            font_path = path
            break
    
    if font_path is None:
        candidates = glob.glob('/usr/share/fonts/**/DejaVuSans.ttf', recursive=True)
        if candidates:
            font_path = candidates[0]
    
    if font_path:
        pdfmetrics.registerFont(TTFont('DejaVu', font_path))
        FONT_NAME = 'DejaVu'
        FONT_BOLD = 'DejaVu'
    else:
        FONT_NAME = 'Helvetica'
        FONT_BOLD = 'Helvetica-Bold'
    
    doc = SimpleDocTemplate(output_path, pagesize=landscape(A4),
                           rightMargin=1.5*cm, leftMargin=1.5*cm,
                           topMargin=1.5*cm, bottomMargin=1.5*cm)
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('T', fontName=FONT_BOLD, fontSize=18, spaceAfter=15, alignment=1)
    heading_style = ParagraphStyle('H', fontName=FONT_BOLD, fontSize=13, spaceBefore=12, spaceAfter=8)
    normal_style = ParagraphStyle('N', fontName=FONT_NAME, fontSize=9, spaceAfter=5)
    
    story.append(Paragraph(f'Анализ руды: {img_name}', title_style))
    story.append(Paragraph(f'Дата: {datetime.now().strftime("%d.%m.%Y %H:%M")}', normal_style))
    story.append(Spacer(1, 0.3*cm))
    
    # Статистика
    class_counts = {}
    class_areas = {}
    for seg in segments:
        cls = seg['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
        class_areas[cls] = class_areas.get(cls, 0) + seg['area_px']
    
    total_image_area = original_np.shape[0] * original_np.shape[1]
    
    # 1. Статистика по классам
    story.append(Paragraph('1. Статистика по классам', heading_style))
    
    table_data = [['Класс', 'Кол-во', 'Площадь (px²)', 'Доля (%)']]
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        area = class_areas[cls]
        fraction = 100 * class_areas[cls] / total_image_area if total_image_area > 0 else 0
        table_data.append([cls, str(count), f'{area:.1f}', f'{fraction:.3f}'])
    
    table = Table(table_data, colWidths=[5*cm, 2*cm, 3.5*cm, 2*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*cm))
    
    # 2. Статистика по группам
    CLASS_GROUPS = {
        'sulfides': ['sulfide_massive', 'sulfide_disseminated'],
        'oxides': ['magnetite', 'other_oxide'],
        'talc': ['talc'],
        'matrix': ['silicate_matrix', 'serpentinite'],
        'structural': ['fracture', 'void'],
    }
    
    story.append(Paragraph('2. Статистика по группам', heading_style))
    
    group_data = [['Группа', 'Кол-во', 'Площадь (px²)', 'Доля (%)']]
    for group_name, group_classes in CLASS_GROUPS.items():
        total_count = sum(class_counts.get(cls, 0) for cls in group_classes)
        total_area_px = sum(class_areas.get(cls, 0) for cls in group_classes)
        fraction = 100 * total_area_px / total_image_area if total_image_area > 0 else 0
        group_data.append([group_name, str(total_count), f'{total_area_px:.1f}', f'{fraction:.3f}'])
    
    group_table = Table(group_data, colWidths=[5*cm, 2*cm, 3.5*cm, 2*cm])
    group_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#548235')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(group_table)
    story.append(Spacer(1, 0.3*cm))
    
    # 3. Графики
    story.append(Paragraph('3. Графики распределения фаз', heading_style))
    
    analysis_classes = [c for c in class_counts.keys()]
    plot_colors = plt.cm.Set2(np.linspace(0, 1, max(len(analysis_classes), 1)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if len(analysis_classes) > 0:
        fractions = [100 * class_areas[cls] / total_image_area if total_image_area > 0 else 0 
                    for cls in analysis_classes]
        bars = ax1.barh(analysis_classes, fractions, color=plot_colors[:len(analysis_classes)], 
                       edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Доля площади, %', fontsize=10)
        ax1.set_title('Доли фаз на изображении', fontsize=11, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars, fractions):
            if val > 0:
                ax1.text(val + max(fractions)*0.02, bar.get_y() + bar.get_height()/2, 
                        f'{val:.2f}%', va='center', fontsize=8)
        
        areas = [class_areas[cls] for cls in analysis_classes]
        non_zero = [(areas[i], analysis_classes[i], plot_colors[i]) 
                   for i in range(len(analysis_classes)) if areas[i] > 0]
        if non_zero:
            vals, names, cols = zip(*non_zero)
            ax2.pie(vals, labels=names, autopct='%1.1f%%', 
                   colors=cols, startangle=90, textprops={'fontsize': 8})
        ax2.set_title('Распределение площадей', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    charts_path = os.path.join(os.path.dirname(output_path), 'temp_charts.png')
    plt.savefig(charts_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    story.append(RLImage(charts_path, width=24*cm, height=8.5*cm))
    story.append(Spacer(1, 0.3*cm))
    
    # 4. Визуализация с масками и ЛЕГЕНДОЙ
    story.append(Paragraph('4. Визуализация с масками', heading_style))
    
    CLASS_COLORS = {
        'sulfide_massive': (1.0, 0.9, 0.3),
        'sulfide_disseminated': (1.0, 0.6, 0.0),
        'magnetite': (0.2, 0.2, 0.2),
        'other_oxide': (0.4, 0.3, 0.3),
        'talc': (0.2, 0.6, 0.3),
        'silicate_matrix': (0.8, 0.8, 0.8),
        'serpentinite': (0.4, 0.6, 0.4),
        'fracture': (1.0, 0.0, 0.0),
        'void': (0.0, 0.0, 0.0),
    }
    
    h_orig, w_orig = original_np.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    colored_mask = np.zeros((h_orig, w_orig, 4), dtype=np.float32)
    for seg in segments:
        mask = seg['mask']
        cls = seg['class']
        color = CLASS_COLORS.get(cls, (0.5, 0.5, 0.5))
        
        if mask.shape != (h_orig, w_orig):
            mask_resized = cv2.resize(mask.astype(np.uint8), (w_orig, h_orig), 
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask_resized = mask
        
        colored_mask[mask_resized, :3] = color
        colored_mask[mask_resized, 3] = 0.5
    
    ax.imshow(original_np)
    ax.imshow(colored_mask)
    ax.set_title(f'Сегментация ({len(segments)} объектов)', fontsize=13)
    ax.axis('off')
    
    # === ДОБАВЛЕНИЕ ЛЕГЕНДЫ ===
    legend_elements = []
    # Добавляем в легенду только те классы, которые реально найдены на фото
    for cls in sorted(class_counts.keys()):
        color = CLASS_COLORS.get(cls, (0.5, 0.5, 0.5))
        count = class_counts[cls]
        legend_elements.append(
            Patch(facecolor=color, label=f'{cls} ({count})', 
                 edgecolor='black', linewidth=0.5)
        )
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=8, framealpha=0.9, ncol=2)
    # ==============================
    
    viz_path = os.path.join(os.path.dirname(output_path), 'temp_viz.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    story.append(RLImage(viz_path, width=20*cm, height=14*cm))
    story.append(Spacer(1, 0.3*cm))
    
    # 5. Анализ замещения и ВИЗУАЛИЗАЦИЯ СРАСТАНИЙ
    story.append(Paragraph('5. Анализ замещения сульфидов', heading_style))
    
    sulfides = [s for s in segments if s['class'] in ['sulfide_massive', 'sulfide_disseminated']]
    
    replacement_types = {'normal': 0, 'replaced': 0, 'intermediate': 0}
    sulfide_details = [] # Для визуализации
    
    for sulfide in sulfides:
        mask = sulfide['mask']
        area = sulfide['area_px']
        
        if mask.shape != (h_orig, w_orig):
            mask_resized = cv2.resize(mask.astype(np.uint8), (w_orig, h_orig), 
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask_resized = mask
        
        ys, xs = np.where(mask_resized)
        if len(xs) == 0:
            continue
        
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(original_np.shape[1], x2 + pad)
        y2 = min(original_np.shape[0], y2 + pad)
        
        patch = original_np[y1:y2, x1:x2]
        patch_mask = mask_resized[y1:y2, x1:x2]
        
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        gray_masked = gray_patch[patch_mask]
        
        dark_ratio = np.sum(gray_masked < 80) / len(gray_masked) if len(gray_masked) > 0 else 0
        
        if dark_ratio < 0.15 and area > 1000:
            rtype = 'normal'
        elif dark_ratio > 0.35 or (area < 1000 and dark_ratio > 0.20):
            rtype = 'replaced'
        else:
            rtype = 'intermediate'
        
        replacement_types[rtype] += 1
        
        sulfide_details.append({
            'patch': patch,
            'patch_mask': patch_mask,
            'area': area,
            'dark_ratio': dark_ratio,
            'type': rtype,
            'confidence': sulfide['confidence']
        })
    
    total_sulfides = sum(replacement_types.values())
    
    replacement_table = [
        ['Тип срастания', 'Кол-во', 'Доля (%)'],
        ['Обычные (рядовая руда)', str(replacement_types['normal']),
         f'{100*replacement_types["normal"]/total_sulfides:.1f}' if total_sulfides > 0 else '0'],
        ['Тонкие (труднообогатимая)', str(replacement_types['replaced']),
         f'{100*replacement_types["replaced"]/total_sulfides:.1f}' if total_sulfides > 0 else '0'],
        ['Промежуточные', str(replacement_types['intermediate']),
         f'{100*replacement_types["intermediate"]/total_sulfides:.1f}' if total_sulfides > 0 else '0'],
    ]
    
    rep_table = Table(replacement_table, colWidths=[7*cm, 2.5*cm, 2.5*cm])
    rep_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#C00000')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(rep_table)
    story.append(Spacer(1, 0.3*cm))
    
    # === ВИЗУАЛИЗАЦИЯ СРАСТАНИЙ ===
    if len(sulfide_details) > 0:
        story.append(Paragraph('Визуализация срастаний', heading_style))
        
        REPL_COLORS = {
            'normal': (0.2, 0.8, 0.2),
            'replaced': (0.9, 0.1, 0.1),
            'intermediate': (1.0, 0.6, 0.0)
        }
        REPL_LABELS = {
            'normal': 'Обычное',
            'replaced': 'Замещённое',
            'intermediate': 'Промежуточное'
        }
        
        # Берём до 6 примеров (по 2 каждого типа)
        examples = []
        for rtype in ['normal', 'replaced', 'intermediate']:
            type_examples = [s for s in sulfide_details if s['type'] == rtype]
            type_examples.sort(key=lambda x: x['area'], reverse=True)
            examples.extend(type_examples[:2])
        
        n_cols = min(3, len(examples))
        n_rows = (len(examples) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, ex in enumerate(examples):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            patch = ex['patch']
            patch_mask = ex['patch_mask']
            
            ax.imshow(patch)
            
            overlay = np.zeros((*patch.shape[:2], 4), dtype=np.float32)
            color = REPL_COLORS[ex['type']]
            overlay[patch_mask, :3] = color
            overlay[patch_mask, 3] = 0.45
            ax.imshow(overlay)
            
            label = REPL_LABELS[ex['type']]
            ax.set_title(
                f"{label}\nПлощадь: {ex['area']} px\nЗамещение: {ex['dark_ratio']:.0%}",
                fontsize=9, fontweight='bold',
                color='green' if ex['type'] == 'normal' else 
                      ('red' if ex['type'] == 'replaced' else 'orange')
            )
            ax.axis('off')
        
        # Скрываем пустые subplot
        for idx in range(len(examples), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        repl_path = os.path.join(os.path.dirname(output_path), 'temp_replacement.png')
        plt.savefig(repl_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        story.append(RLImage(repl_path, width=22*cm, height=7*cm))
        story.append(Spacer(1, 0.3*cm))
    
    # 6. Итоговая сводка
    story.append(Paragraph('6. Итоговая сводка', heading_style))
    
    talc_area = class_areas.get('talc', 0)
    talc_fraction = 100 * talc_area / total_image_area if total_image_area > 0 else 0
    
    if total_sulfides > 0:
        replacement_index = (replacement_types['replaced'] / total_sulfides) * 0.7 + (talc_fraction / 10) * 0.3
    else:
        replacement_index = 0
    
    if replacement_index < 0.3:
        ore_type = 'ЛЕГКООБОГАТИМАЯ'
        recommendation = 'Стандартная флотационная схема'
    elif replacement_index < 0.6:
        ore_type = 'СРЕДНЕЙ ОБОГАТИМОСТИ'
        recommendation = 'Тонкое измельчение, комбинированные схемы'
    else:
        ore_type = 'ТРУДНООБОГАТИМАЯ'
        recommendation = 'Селективная флотация с депрессорами'
    
    for line in [
        f'<b>Всего сульфидов:</b> {total_sulfides}',
        f'<b>Доля талька:</b> {talc_fraction:.3f}%',
        f'<b>Индекс труднообогатимости:</b> {replacement_index:.3f}',
        f'<b>Тип руды:</b> {ore_type}',
        f'<b>Рекомендация:</b> {recommendation}',
    ]:
        story.append(Paragraph(line, normal_style))
    
    doc.build(story)
    
    # Очистка временных файлов
    for f in [charts_path, viz_path, repl_path]:
        if os.path.exists(f):
            os.remove(f)

# ============================================
# ИНТЕРФЕЙС STREAMLIT
# ============================================

st.title('🔬 Анализ руд')
st.markdown('**Пошаговый анализ микроструктур с генерацией PDF отчётов**')

# Боковая панель
with st.sidebar:
    st.header('⚙️ Настройки')
    
    st.subheader('Предобработка')
    clahe_enabled = st.checkbox('CLAHE (контраст)', value=True)
    denoise_enabled = st.checkbox('Шумоподавление', value=True)
    max_image_size = st.slider('Макс. размер (px)', 0, 4096, 2048, step=256)
    
    st.subheader('Сегментация')
    min_segment_area = st.slider('Мин. площадь сегмента', 100, 1000, 300, step=50)
    
    config = {
        'clahe_enabled': clahe_enabled,
        'denoise_enabled': denoise_enabled,
        'max_image_size': max_image_size,
        'min_segment_area': min_segment_area,
    }

# ШАГ 1: ПРОВЕРКА БИБЛИОТЕК
if st.session_state.step == 1:
    st.header('📋 Шаг 1: Проверка библиотек')
    
    if st.button('🔍 Проверить библиотеки', type='primary'):
        with st.spinner('Проверка...'):
            libs = check_libraries()
            st.session_state.libraries = libs
            
            all_installed = all(lib['status'] == '✅' for lib in libs.values())
            
            if all_installed:
                st.success('✅ Все необходимые библиотеки установлены!')
            else:
                st.warning('⚠️ Некоторые библиотеки отсутствуют')
    
    if 'libraries' in st.session_state:
        st.subheader('Результаты проверки:')
        
        libs = st.session_state.libraries
        cols = st.columns(3)
        
        for i, (name, info) in enumerate(libs.items()):
            with cols[i % 3]:
                if info['status'] == '✅':
                    st.success(f'{info["status"]} {name}: {info["version"]}')
                else:
                    st.error(f'{info["status"]} {name}: {info["version"]}')
        
        all_installed = all(lib['status'] == '✅' for lib in libs.values())
        
        if all_installed:
            if st.button('Перейти к загрузке изображений ➡️', type='primary'):
                st.session_state.step = 2
                st.rerun()

# ШАГ 2: ЗАГРУЗКА ИЗОБРАЖЕНИЙ
elif st.session_state.step == 2:
    st.header('📁 Шаг 2: Загрузка изображений')
    
    uploaded_files = st.file_uploader(
        'Загрузите фотографии руды (TIFF, PNG, JPEG)',
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        st.success(f'✅ Загружено файлов: {len(uploaded_files)}')
        
        st.subheader('Загруженные файлы:')
        for f in uploaded_files:
            st.write(f'📄 {f.name} ({f.size / 1024:.1f} KB)')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('◀️ Назад к проверке'):
                st.session_state.step = 1
                st.rerun()
        
        with col2:
            if st.button('Запустить обработку 🚀', type='primary'):
                st.session_state.step = 3
                st.rerun()

# ШАГ 3: ОБРАБОТКА
elif st.session_state.step == 3:
    st.header('⚡ Шаг 3: Обработка изображений')
    
    if not st.session_state.uploaded_files:
        st.warning('⚠️ Нет загруженных файлов!')
        if st.button('◀️ Вернуться к загрузке'):
            st.session_state.step = 2
            st.rerun()
    else:
        # Загрузка моделей
        if not st.session_state.models_loaded:
            with st.spinner('📥 Загрузка моделей...'):
                try:
                    mask_generator, clip_model, clip_preprocess, device = load_models()
                    st.session_state.models_loaded = True
                    st.session_state.mask_generator = mask_generator
                    st.session_state.clip_model = clip_model
                    st.session_state.clip_preprocess = clip_preprocess
                    st.session_state.device = device
                    st.success('✅ Модели загружены!')
                except Exception as e:
                    st.error(f'❌ Ошибка загрузки моделей: {e}')
                    st.stop()
        
        st.write(f'🖼 Всего изображений: {len(st.session_state.uploaded_files)}')
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(st.session_state.uploaded_files):
            status_text.text(f'Обработка {idx + 1}/{len(st.session_state.uploaded_files)}: {uploaded_file.name}')
            
            try:
                segments, original_np, preprocess_info = process_image(
                    uploaded_file,
                    st.session_state.mask_generator,
                    st.session_state.clip_model,
                    st.session_state.clip_preprocess,
                    st.session_state.device,
                    config
                )
                
                class_counts = {}
                class_areas = {}
                for seg in segments:
                    cls = seg['class']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                    class_areas[cls] = class_areas.get(cls, 0) + seg['area_px']
                
                pdf_path = os.path.join(st.session_state.pdf_dir, f'{os.path.splitext(uploaded_file.name)[0]}_report.pdf')
                generate_pdf(segments, original_np, uploaded_file.name, pdf_path)
                
                st.session_state.processing_results[uploaded_file.name] = {
                    'segments': segments,
                    'original_np': original_np,
                    'class_counts': class_counts,
                    'class_areas': class_areas,
                    'pdf_path': pdf_path,
                    'total_objects': len(segments),
                }
                
                progress_bar.progress((idx + 1) / len(st.session_state.uploaded_files))
            
            except Exception as e:
                st.error(f'❌ Ошибка обработки {uploaded_file.name}: {e}')
                import traceback
                st.code(traceback.format_exc())
        
        status_text.text('✅ Обработка завершена!')
        progress_bar.empty()
        
        st.success(f'✅ Обработано {len(st.session_state.processing_results)} изображений')
        
        st.subheader('📊 Результаты обработки:')
        
        for filename, result in st.session_state.processing_results.items():
            with st.expander(f' {filename} ({result["total_objects"]} объектов)', expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write('**Статистика по классам:**')
                    
                    table_data = []
                    for cls, count in result['class_counts'].items():
                        area = result['class_areas'].get(cls, 0)
                        table_data.append([cls, count, f'{area:.0f} px²'])
                    
                    st.table(pd.DataFrame(table_data, columns=['Класс', 'Количество', 'Площадь']))
                
                with col2:
                    CLASS_COLORS_UI = {
                        'sulfide_massive': (1.0, 0.9, 0.3),
                        'sulfide_disseminated': (1.0, 0.6, 0.0),
                        'magnetite': (0.2, 0.2, 0.2),
                        'talc': (0.2, 0.6, 0.3),
                        'fracture': (1.0, 0.0, 0.0),
                        'void': (0.0, 0.0, 0.0),
                    }
                    
                    h_orig, w_orig = result['original_np'].shape[:2]
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    
                    colored_mask = np.zeros((h_orig, w_orig, 4), dtype=np.float32)
                    for seg in result['segments']:
                        mask = seg['mask']
                        cls = seg['class']
                        color = CLASS_COLORS_UI.get(cls, (0.5, 0.5, 0.5))
                        
                        if mask.shape != (h_orig, w_orig):
                            mask_resized = cv2.resize(mask.astype(np.uint8), (w_orig, h_orig),
                                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                        else:
                            mask_resized = mask
                        
                        colored_mask[mask_resized, :3] = color
                        colored_mask[mask_resized, 3] = 0.5
                    
                    ax.imshow(result['original_np'])
                    ax.imshow(colored_mask)
                    ax.set_title(f'{result["total_objects"]} объектов')
                    ax.axis('off')
                    
                    st.pyplot(fig, use_container_width=True)
                
                with open(result['pdf_path'], 'rb') as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                st.download_button(
                    label='📥 Скачать PDF отчёт',
                    data=pdf_bytes,
                    file_name=f'{os.path.splitext(filename)[0]}_report.pdf',
                    mime='application/pdf',
                    type='primary'
                )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('◀️ Загрузить новые файлы'):
                st.session_state.uploaded_files = []
                st.session_state.processing_results = {}
                st.session_state.step = 2
                st.rerun()
        
        with col2:
            if st.button('🔄 Начать заново'):
                st.session_state.uploaded_files = []
                st.session_state.processing_results = {}
                st.session_state.models_loaded = False
                st.session_state.step = 1
                st.rerun()

# Очистка временных файлов при закрытии
def cleanup():
    if os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)

atexit.register(cleanup)
