import os
import json
import base64
import io
import numpy as np
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import xarray as xr
except ImportError:
    xr = None

# We store the HTML template as a large string
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>LandTrendr Validation</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script>tailwind.config = { darkMode: 'class' }</script>
    <style>
        .pixelated { image-rendering: pixelated; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
        .point-item { cursor: pointer; border-left: 4px solid transparent; }
        .point-item:hover { background-color: #334155; }
        .active-point { background-color: #1e293b; border-left-color: #3b82f6; }
        
        input[type=range] { -webkit-appearance: none; width: 100%; background: transparent; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; height: 20px; width: 20px; border-radius: 50%; background: #3b82f6; cursor: pointer; margin-top: -6px; }
        input[type=range]::-webkit-slider-runnable-track { width: 100%; height: 8px; cursor: pointer; background: #475569; border-radius: 4px; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 font-sans h-screen flex flex-col overflow-hidden">
    
    <header class="bg-gray-950 p-3 shadow-md flex justify-between items-center z-20 shrink-0 border-b border-gray-800">
        <div class="flex items-center gap-2 lg:gap-6">
            <h1 class="text-sm lg:text-xl font-bold">LandTrendr Validation</h1>
            <div class="flex gap-2 text-[10px] lg:text-sm">
                <div class="bg-gray-800 px-2 py-1 rounded">Acc: <span id="acc-val" class="font-bold text-green-400">0%</span></div>
                <div class="bg-gray-800 px-2 py-1 rounded">Kappa: <span id="kappa-val" class="font-bold text-blue-400">0.0</span></div>
            </div>
        </div>
        <div class="flex items-center gap-3 lg:gap-6">
            <div class="text-xs lg:text-sm hidden sm:block">Progress: <span id="progress-text" class="font-bold">0/0</span></div>
            <button onclick="exportCSV()" class="bg-green-600 hover:bg-green-500 text-white px-3 py-1.5 rounded font-bold shadow text-xs lg:text-sm flex items-center gap-2">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                <span class="hidden sm:inline">Export CSV</span>
            </button>
        </div>
    </header>

    <div class="flex flex-1 overflow-hidden min-h-0">
        <div class="hidden lg:flex flex-col w-1/4 bg-gray-800 border-r border-gray-700 shrink-0 h-full z-10">
            <div class="p-4 bg-gray-900 border-b border-gray-700 font-bold text-lg">Validation Points</div>
            <div class="overflow-y-auto flex-1 p-2 flex flex-col gap-1" id="point-list"></div>
        </div>

        <div class="flex flex-col flex-1 p-2 lg:p-6 overflow-y-auto lg:overflow-hidden min-h-0 bg-gray-900 gap-2 lg:gap-6">
            <div class="flex lg:hidden flex-col bg-gray-800 p-3 rounded-lg border border-gray-700 shadow shrink-0 gap-3">
                <div class="flex justify-between items-center">
                    <button onclick="prevPoint()" class="bg-gray-700 px-3 py-1 rounded text-white shadow font-bold text-sm">&larr; Prev</button>
                    <div class="text-center">
                        <div class="font-bold text-sm" id="mob-point-id">VAL-000</div>
                        <div class="text-[10px] text-gray-400">Pred: <span id="mob-pt-pred" class="text-blue-400"></span></div>
                    </div>
                    <button onclick="nextPoint()" class="bg-gray-700 px-3 py-1 rounded text-white shadow font-bold text-sm">Next &rarr;</button>
                </div>
                <div class="flex justify-between items-center gap-2">
                    <select id="val-mob" class="flex-1 border border-gray-600 rounded px-2 py-2 bg-gray-700 text-white font-bold text-sm" onchange="syncSelects('mob')"></select>
                    <button onclick="saveAndNext()" class="bg-blue-600 px-4 py-2 rounded text-white font-bold text-sm">Save</button>
                </div>
                <div class="flex flex-col gap-1 mt-1">
                    <div class="flex justify-between text-[10px] text-gray-400 font-bold px-1">
                        <span id="slider-min-display">2000</span>
                        <span id="slider-val" class="text-blue-400">2010</span>
                        <span id="slider-max-display">2023</span>
                    </div>
                    <input type="range" id="year-slider" min="0" max="23" value="0" oninput="onSliderMove(this.value)">
                </div>
            </div>

            <div class="hidden lg:flex justify-between items-center bg-gray-800 p-5 rounded-lg border border-gray-700 shadow shrink-0">
                <div class="flex items-center gap-6">
                    <div>
                        <h2 class="text-2xl font-bold" id="desk-point-id">Point VAL-000</h2>
                        <p class="text-gray-400 text-sm">Lat: <span id="desk-pt-lat"></span> | Lon: <span id="desk-pt-lon"></span></p>
                    </div>
                    <div class="h-10 border-l border-gray-600"></div>
                    <div class="text-center">
                        <div class="text-sm text-gray-400">Predicted YOD</div>
                        <div class="text-2xl font-bold text-blue-400" id="desk-pt-pred"></div>
                    </div>
                </div>
                <div class="flex items-center gap-4">
                    <label class="font-bold text-gray-300">Expert Validation:</label>
                    <select id="val-desk" class="border border-gray-600 rounded px-4 py-2 bg-gray-700 text-white font-bold text-lg" onchange="syncSelects('desk')"></select>
                    <button onclick="saveAndNext()" class="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded text-white font-bold shadow text-lg transition-colors">
                        Save & Next
                    </button>
                </div>
            </div>

            <div class="flex flex-col lg:flex-row flex-1 gap-2 lg:gap-6 min-h-0 pb-10 lg:pb-0">
                <div class="flex-1 bg-gray-800 rounded-lg border border-gray-700 p-1 lg:p-4 flex flex-col min-h-[250px] lg:min-h-0 relative">
                    <h3 class="hidden lg:block font-bold text-gray-300 border-b border-gray-700 pb-2 mb-2">Temporal Trajectory</h3>
                    <div id="plotly-div" class="w-full h-full lg:flex-1 absolute lg:relative inset-0 p-1 lg:p-0"></div>
                </div>
                
                <div class="w-full lg:w-[400px] bg-gray-800 rounded-lg border border-gray-700 p-2 lg:p-6 flex flex-row lg:flex-col items-center justify-center lg:justify-start gap-4 shrink-0">
                    <div class="hidden lg:flex justify-between w-full border-b border-gray-700 pb-4 mb-4">
                        <h3 class="font-bold text-gray-300">True Color Context</h3>
                        <span id="anim-year-desk" class="font-bold bg-blue-900 text-blue-300 px-3 py-1 rounded-full text-lg">--</span>
                    </div>
                    
                    <div class="relative w-32 h-32 lg:w-72 lg:h-72 shrink-0">
                        <img id="anim-img" src="" class="w-full h-full object-cover pixelated border-4 border-transparent rounded shadow-inner">
                        <div id="anim-year-mob" class="lg:hidden absolute bottom-1 right-1 bg-black/70 text-white px-2 py-0.5 rounded text-xs font-bold">--</div>
                    </div>
                    
                    <div class="flex flex-col text-xs lg:text-sm text-gray-400 gap-1 w-full text-left lg:text-center mt-0 lg:mt-6">
                        <div class="lg:hidden font-bold text-gray-300 border-b border-gray-700 pb-1 mb-1">Context</div>
                        <p class="hidden lg:block">The cross indicates the pixel center being analyzed. Drag timeline or hover graph to explore.</p>
                        <div class="lg:hidden">Pred: <span id="img-pred-val-mob" class="font-bold text-red-400"></span></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const DASHBOARD_DATA = /*INJECT_DATA_HERE*/{};
        window.currentActiveIndex = 0;
        
        const listDiv = document.getElementById('point-list');
        const plotlyDiv = document.getElementById('plotly-div');
        const selectMob = document.getElementById('val-mob');
        const selectDesk = document.getElementById('val-desk');
        const slider = document.getElementById('year-slider');
        
        slider.min = 0;
        slider.max = DASHBOARD_DATA.years.length - 1;
        document.getElementById('slider-min-display').innerText = DASHBOARD_DATA.years[0];
        document.getElementById('slider-max-display').innerText = DASHBOARD_DATA.years[DASHBOARD_DATA.years.length-1];

        const optsHTML = `<option value="None">No Disturbance</option>` + 
                         DASHBOARD_DATA.years.map(y => `<option value="${y}">${y}</option>`).join('');
        selectMob.innerHTML = optsHTML;
        selectDesk.innerHTML = optsHTML;

        function syncSelects(source) {
            if(source === 'mob') selectDesk.value = selectMob.value;
            if(source === 'desk') selectMob.value = selectDesk.value;
        }

        function calculateMetrics() {
            const validated = DASHBOARD_DATA.points.filter(p => p.validated_val !== null);
            document.getElementById('progress-text').innerText = `${validated.length}/${DASHBOARD_DATA.points.length}`;

            if (validated.length === 0) {
                document.getElementById('acc-val').innerText = '0%';
                document.getElementById('kappa-val').innerText = '0.00';
                return;
            }
            
            let correct = 0; let trueCounts = {}; let predCounts = {}; let classes = new Set();
            validated.forEach(p => {
                let t = String(p.validated_val); let pr = String(p.pred_val || "None");
                classes.add(t); classes.add(pr);
                if (t === pr) correct++;
                trueCounts[t] = (trueCounts[t] || 0) + 1; predCounts[pr] = (predCounts[pr] || 0) + 1;
            });
            const po = correct / validated.length; let pe = 0;
            classes.forEach(c => { pe += ((trueCounts[c] || 0)/validated.length) * ((predCounts[c] || 0)/validated.length); });
            let kappa = (po - pe) / (1 - pe); if (isNaN(kappa) || !isFinite(kappa)) kappa = 0;

            document.getElementById('acc-val').innerText = (po * 100).toFixed(0) + '%';
            document.getElementById('kappa-val').innerText = kappa.toFixed(2);
        }

        function buildList() {
            listDiv.innerHTML = '';
            DASHBOARD_DATA.points.forEach((pt, idx) => {
                const div = document.createElement('div');
                div.id = 'item-' + idx;
                
                let badgeHTML = pt.validated_val === null 
                    ? `<span class="bg-yellow-900/50 text-yellow-300 text-xs px-2 py-0.5 rounded font-bold shadow-sm">Pend</span>`
                    : (String(pt.validated_val) === String(pt.pred_val || "None") 
                        ? `<span class="bg-green-900/50 text-green-300 text-xs px-2 py-0.5 rounded font-bold shadow-sm">Match</span>` 
                        : `<span class="bg-red-900/50 text-red-300 text-xs px-2 py-0.5 rounded font-bold shadow-sm">Err</span>`);

                div.className = `point-item p-3 border-b border-gray-700 flex justify-between items-center rounded-lg ${idx === window.currentActiveIndex ? 'active-point' : ''}`;
                div.innerHTML = `<div class="text-sm font-bold text-gray-200">${pt.id}</div><div>${badgeHTML}</div>`;
                div.onclick = () => loadPoint(idx);
                listDiv.appendChild(div);
            });
            calculateMetrics();
        }

        function onSliderMove(idxStr) {
            const idx = parseInt(idxStr);
            const pt = DASHBOARD_DATA.points[window.currentActiveIndex];
            const year = DASHBOARD_DATA.years[idx];
            
            const imgEl = document.getElementById('anim-img');
            imgEl.src = "data:image/png;base64," + pt.images[idx];
            document.getElementById('anim-year-desk').innerText = year;
            document.getElementById('anim-year-mob').innerText = year;
            document.getElementById('slider-val').innerText = year;
            
            if (year === pt.pred_val) {
                imgEl.classList.remove('border-transparent');
                imgEl.classList.add('border-red-500');
            } else {
                imgEl.classList.remove('border-red-500');
                imgEl.classList.add('border-transparent');
            }

            if(plotlyDiv.layout && plotlyDiv.layout.shapes) {
                Plotly.relayout(plotlyDiv, { 'shapes[0].x0': year, 'shapes[0].x1': year });
            }
        }

        function updateImageOnly(year) {
            const pt = DASHBOARD_DATA.points[window.currentActiveIndex];
            const idx = DASHBOARD_DATA.years.indexOf(year);
            if(idx !== -1) {
                slider.value = idx; 
                document.getElementById('slider-val').innerText = year;
                const imgEl = document.getElementById('anim-img');
                imgEl.src = "data:image/png;base64," + pt.images[idx];
                document.getElementById('anim-year-desk').innerText = year;
                document.getElementById('anim-year-mob').innerText = year;
                if (year === pt.pred_val) {
                    imgEl.classList.remove('border-transparent'); imgEl.classList.add('border-red-500');
                } else {
                    imgEl.classList.remove('border-red-500'); imgEl.classList.add('border-transparent');
                }
            }
        }

        function loadPoint(index) {
            if (index < 0 || index >= DASHBOARD_DATA.points.length) return;
            window.currentActiveIndex = index;
            buildList();
            
            const pt = DASHBOARD_DATA.points[index];
            document.getElementById('desk-point-id').innerText = pt.id || ("VAL-"+index);
            document.getElementById('desk-pt-lat').innerText = pt.lat ? pt.lat.toFixed(5) : "N/A";
            document.getElementById('desk-pt-lon').innerText = pt.lon ? pt.lon.toFixed(5) : "N/A";
            document.getElementById('desk-pt-pred').innerText = pt.pred_val || "None";
            document.getElementById('mob-point-id').innerText = pt.id || ("VAL-"+index);
            document.getElementById('mob-pt-pred').innerText = pt.pred_val || "None";
            document.getElementById('img-pred-val-mob').innerText = pt.pred_val || "None";
            
            const valToSet = pt.validated_val !== null ? pt.validated_val : (pt.pred_val || "None");
            selectMob.value = valToSet; selectDesk.value = valToSet;

            const isMobile = window.innerWidth < 1024;
            
            const traces = [];
            
            if (pt.ts_raw && pt.ts_raw.length > 0) {
                traces.push({
                    x: DASHBOARD_DATA.years, y: pt.ts_raw, mode: 'markers+lines', name: 'Raw Index',
                    line: {color: 'rgba(148, 163, 184, 0.5)', width: 2}, marker: {color: '#94a3b8', size: isMobile ? 6 : 8}
                });
            }
            if (pt.fit_vals && pt.fit_vals.length > 0) {
                traces.push({
                    x: pt.fit_years, y: pt.fit_vals, mode: 'lines+markers', name: 'LT Fit',
                    line: {color: '#60a5fa', width: 3}, marker: {color: '#93c5fd', size: isMobile ? 8 : 10, symbol: 'square'}
                });
            }
            
            const initialYear = pt.pred_val || DASHBOARD_DATA.years[0];
            const initialIdx = DASHBOARD_DATA.years.indexOf(initialYear);
            
            const layout = {
                margin: {t: isMobile? 5: 20, l: isMobile? 30: 40, r: isMobile? 5: 20, b: isMobile? 20: 40},
                showlegend: !isMobile,
                legend: {orientation: "h", y: -0.2, font: {color: '#e2e8f0'}},
                paper_bgcolor: '#1f2937', plot_bgcolor: '#1f2937',
                xaxis: { color: '#e2e8f0', gridcolor: '#334155', fixedrange: isMobile },
                yaxis: { color: '#e2e8f0', gridcolor: '#334155', fixedrange: isMobile },
                hovermode: 'x unified', dragmode: isMobile ? false : 'zoom',
                shapes: [{
                    type: 'line', x0: initialYear, y0: 0, x1: initialYear, y1: 1, yref: 'paper',
                    line: { color: 'rgba(239, 68, 68, 0.5)', width: 3, dash: 'dot' }
                }]
            };

            Plotly.react(plotlyDiv, traces, layout, {responsive: true, displayModeBar: !isMobile});
            
            onSliderMove(initialIdx !== -1 ? initialIdx : 0);

            if (!plotlyDiv.hasHoverEvent) {
                plotlyDiv.on('plotly_hover', function(data) {
                    if (data.points && data.points.length > 0) {
                        updateImageOnly(data.points[0].x);
                    }
                });
                plotlyDiv.hasHoverEvent = true;
            }
        }

        function prevPoint() { loadPoint(window.currentActiveIndex - 1); }
        function nextPoint() { loadPoint(window.currentActiveIndex + 1); }

        function saveAndNext() {
            const pt = DASHBOARD_DATA.points[window.currentActiveIndex];
            const val = selectDesk.value; 
            pt.validated_val = val === "None" ? null : parseInt(val);
            if (window.currentActiveIndex < DASHBOARD_DATA.points.length - 1) {
                nextPoint();
            } else {
                buildList(); 
                alert("Validation Complete! Click 'Export CSV'.");
            }
        }

        function exportCSV() {
            let csvContent = "data:text/csv;charset=utf-8,point_id,lat,lon,predicted_year,validated_year\\n";
            DASHBOARD_DATA.points.forEach(pt => {
                let val = pt.validated_val !== null ? pt.validated_val : "";
                let pred = pt.pred_val !== null ? pt.pred_val : "";
                csvContent += `${pt.id},${pt.lat},${pt.lon},${pred},${val}\\n`;
            });
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "cdts_validation_results.csv");
            document.body.appendChild(link); link.click(); document.body.removeChild(link);
        }

        window.addEventListener('resize', function() {
            if (window.currentActiveIndex !== -1) {
                Plotly.Plots.resize(plotlyDiv);
                const isMobile = window.innerWidth < 1024;
                Plotly.relayout(plotlyDiv, {
                    'dragmode': isMobile ? false : 'zoom',
                    'xaxis.fixedrange': isMobile,
                    'yaxis.fixedrange': isMobile,
                    'showlegend': !isMobile
                });
            }
        });

        buildList();
        loadPoint(0);
    </script>
</body>
</html>
"""

def extract_image_chip(arr, center_y, center_x, half_win):
    """
    Extracts a (2*half_win+1) x (2*half_win+1) chip safely.
    arr shape: (Bands, Y, X)
    """
    if arr.ndim != 3:
        raise ValueError(f"Array must be 3D (Bands, Y, X), got {arr.shape}")
        
    _, h, w = arr.shape
    y_min = max(0, center_y - half_win)
    y_max = min(h, center_y + half_win + 1)
    x_min = max(0, center_x - half_win)
    x_max = min(w, center_x + half_win + 1)
    
    chip = arr[:, y_min:y_max, x_min:x_max]
    
    # Pad if near edges to always return window_size x window_size
    pad_top = max(0, -(center_y - half_win))
    pad_bottom = max(0, (center_y + half_win + 1) - h)
    pad_left = max(0, -(center_x - half_win))
    pad_right = max(0, (center_x + half_win + 1) - w)
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        chip = np.pad(chip, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
        
    return chip

def encode_chip_to_base64(chip_rgb, add_cross=True):
    """
    chip_rgb shape: (3, H, W) normalized or 0-255.
    Converts to Base64 PNG.
    """
    # Transpose to (H, W, 3)
    img_arr = np.transpose(chip_rgb, (1, 2, 0))
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    
    if add_cross:
        cy, cx = img_arr.shape[0]//2, img_arr.shape[1]//2
        img_arr[cy, cx-2:cx+3] = [255, 0, 0]
        img_arr[cy-2:cy+3, cx] = [255, 0, 0]
        
    img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def _auto_detect_bands(cube):
    """
    Tries to automatically detect True Color RGB and common index bands.
    """
    band_names = cube.coords['band'].values.tolist() if 'band' in cube.coords else []
    
    # RGB heuristics (True Color)
    rgb = None
    common_rgb_names = [('B04', 'B03', 'B02'), ('red', 'green', 'blue'), ('R', 'G', 'B')]
    for candidates in common_rgb_names:
        if all(b in band_names for b in candidates):
            rgb = list(candidates)
            break
            
    # If not found, use first 3 bands
    if rgb is None and len(band_names) >= 3:
        rgb = band_names[:3]
        
    # Find a reasonable raw band for plotting (like NBR, NDVI)
    raw = None
    common_indices = ['NBR', 'NDVI', 'EVI', 'SWIR', 'B12']
    for b in common_indices:
        if b in band_names:
            raw = b
            break
    if raw is None and len(band_names) > 0:
        raw = band_names[0]
        
    return rgb, raw

def generate_landtrendr_accuracy_dashboard(
    cube, 
    points,
    lt_results=None,
    output_html="lt_accuracy_dashboard.html",
    window_size=25,
    year_dim="time"
):
    """
    Generates a standalone HTML Serverless Validation Dashboard for LandTrendr.
    
    Parameters:
    -----------
    cube : xarray.DataArray
        The raw/stac spatio-temporal data cube. Dimensions must include (time, band, y, x) or similar.
    points : str, geopandas.GeoDataFrame, or list of tuples
        Points to validate. Can be a path to a vector file (e.g. '.shp', '.gpkg'), a GeoDataFrame, 
        or a list of (lon, lat) tuples.
    lt_results : xarray.DataArray or xarray.Dataset, optional
        The output from `cdts.metrics.extract_events` or a LandTrendr process. If provided, 
        the dashboard will automatically read predicted YOD and fitted curves.
    output_html : str
        Path to save the generated HTML file.
    window_size : int
        Size of the true color spatial context window (must be odd). Default 25 (25x25 pixels).
    """
    if Image is None:
        raise ImportError("Pillow (PIL) is required. Install it with `pip install Pillow`.")
        
    # 1. Parse Points
    if isinstance(points, str):
        if gpd is None:
            raise ImportError("geopandas is required to read vector files.")
        gdf = gpd.read_file(points)
        point_coords = [(geom.x, geom.y) for geom in gdf.geometry]
        point_ids = gdf.index.tolist() if 'id' not in gdf.columns else gdf['id'].tolist()
    elif gpd is not None and isinstance(points, gpd.GeoDataFrame):
        point_coords = [(geom.x, geom.y) for geom in points.geometry]
        point_ids = points.index.tolist() if 'id' not in points.columns else points['id'].tolist()
    else:
        point_coords = points
        point_ids = [f"VAL-{i:03d}" for i in range(len(points))]

    # 2. Extract Time Dimension
    try:
        years = cube[year_dim].dt.year.values.tolist()
    except AttributeError:
        years = cube[year_dim].values.tolist()
    years = [int(y) for y in years]
    
    # 3. Auto-detect Bands (True Color and Raw Index)
    rgb_bands, raw_band = _auto_detect_bands(cube)

    dashboard_data = {
        "years": years,
        "points": []
    }
    
    half_win = window_size // 2
    
    # We iterate over points to build the payload
    for i, (lon, lat) in enumerate(point_coords):
        # Find nearest pixel indices in the cube
        y_coord_name = 'y' if 'y' in cube.coords else 'lat'
        x_coord_name = 'x' if 'x' in cube.coords else 'lon'
        
        # Actual integer index for spatial cropping
        try:
            y_idx = np.abs(cube[y_coord_name].values - lat).argmin()
            x_idx = np.abs(cube[x_coord_name].values - lon).argmin()
        except Exception:
            # Fallback if coordinates are messed up
            y_idx, x_idx = 0, 0
            
        # Extract Raw Trajectory
        raw_ts = []
        if raw_band:
            try:
                pixel_ds = cube.sel({x_coord_name: lon, y_coord_name: lat}, method="nearest")
                raw_ts = pixel_ds.sel(band=raw_band).values.tolist()
            except Exception:
                pass
            
        # Extract LandTrendr Metrics (if provided)
        fit_vals = []
        fit_years = years # Fit shares the same temporal axis
        pred_yod = None
        
        if lt_results is not None:
            try:
                lt_pixel = lt_results.sel({x_coord_name: lon, y_coord_name: lat}, method="nearest")
                
                # Check if it's a dataset with variables, or DataArray with bands
                if isinstance(lt_pixel, xr.Dataset):
                    if 'yod' in lt_pixel:
                        yod_val = lt_pixel['yod'].values
                        if not np.isnan(yod_val) and yod_val > 0:
                            pred_yod = int(yod_val)
                    # We might not have fit_vals in extract_events directly, but if they exist:
                    if 'fit' in lt_pixel:
                        fit_vals = lt_pixel['fit'].values.tolist()
                        
                elif isinstance(lt_pixel, xr.DataArray):
                    bands = lt_pixel.coords['band'].values.tolist()
                    if 'yod' in bands:
                        yod_val = lt_pixel.sel(band='yod').values
                        if not np.isnan(yod_val) and yod_val > 0:
                            pred_yod = int(yod_val)
            except Exception:
                pass
                
        # 4. Extract True Color Context Images for each year
        images_b64 = []
        for t_idx, year in enumerate(years):
            arr_t = None
            if rgb_bands is not None:
                try:
                    arr_t = cube.isel({year_dim: t_idx}).sel(band=rgb_bands).values
                except KeyError:
                    pass
            
            # Fallback if specific bands fail
            if arr_t is None:
                arr_t = cube.isel({year_dim: t_idx}).values
                if arr_t.shape[0] >= 3:
                    arr_t = arr_t[:3]
                else:
                    arr_t = np.repeat(arr_t[0:1], 3, axis=0)
            
            # Normalize for visualization (Simple Stretch)
            p2, p98 = np.nanpercentile(arr_t, (2, 98))
            if p98 > p2:
                arr_t_norm = (arr_t - p2) / (p98 - p2) * 255.0
            else:
                arr_t_norm = arr_t * 0.0 # pure black if invalid
                
            chip = extract_image_chip(arr_t_norm, y_idx, x_idx, half_win)
            b64_str = encode_chip_to_base64(chip, add_cross=True)
            images_b64.append(b64_str)
            
        dashboard_data["points"].append({
            "id": str(point_ids[i]),
            "lat": float(lat),
            "lon": float(lon),
            "pred_val": pred_yod,
            "validated_val": None,
            "ts_raw": raw_ts,
            "fit_years": fit_years,
            "fit_vals": fit_vals,
            "images": images_b64
        })

    # Inject JSON into HTML
    final_html = HTML_TEMPLATE.replace("/*INJECT_DATA_HERE*/{}", json.dumps(dashboard_data))
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"✅ LandTrendr Accuracy Dashboard generated at: {output_html}")
    return output_html
