/* Single Page Routing & Frontend Logic */

document.addEventListener("DOMContentLoaded", () => {
    // API config
    const API_BASE = "http://localhost:8000/api";
    
    // 12 Cell Types Metadata
    const CELL_TYPES_DATA = {
        "BA":  { name: "Basophil", full_name: "Bạch cầu ái kiềm (Basophil)", color: "#6C5CE7", description: "Bạch cầu hạt có hạt bắt màu kiềm đậm, tham gia phản ứng dị ứng." },
        "BNE": { name: "Band Neutrophil", full_name: "Bạch cầu đoạn trung tính dạng đũa", color: "#0984E3", description: "Bạch cầu trung tính chưa phân đoạn hoàn toàn, nhân hình chữ U hoặc đũa." },
        "EO":  { name: "Eosinophil", full_name: "Bạch cầu ái toan (Eosinophil)", color: "#E17055", description: "Bạch cầu hạt có các hạt bắt màu acid cam đỏ, chống ký sinh trùng." },
        "ERB": { name: "Erythroblast", full_name: "Tiền hồng cầu (Erythroblast)", color: "#D63031", description: "Tế bào tiền thân hồng cầu có nhân trong tủy xương." },
        "LY":  { name: "Lymphocyte", full_name: "Bạch cầu Lympho (Lymphocyte)", color: "#00B894", description: "Tế bào miễn dịch chủ chốt, nhân tròn lớn chiếm gần hết bào tương." },
        "MMY": { name: "Metamyelocyte", full_name: "Hậu tủy bào (Metamyelocyte)", color: "#FDCB6E", description: "Tế bào tủy dòng hạt giai đoạn sau, nhân bắt đầu lõm hình hạt đậu." },
        "MO":  { name: "Monocyte", full_name: "Bạch cầu Mono (Monocyte)", color: "#E84393", description: "Bạch cầu kích thước lớn nhất, nhân hình hạt đậu hoặc uốn khúc." },
        "MY":  { name: "Myelocyte", full_name: "Tủy bào (Myelocyte)", color: "#00CEC9", description: "Tế bào tủy dòng hạt giai đoạn trung gian, nhân tròn hoặc bầu dục." },
        "MYO": { name: "Myeloblast", full_name: "Nguyên tủy bào (Myeloblast)", color: "#A29BFE", description: "Tế bào gốc non nhất dòng tủy, kích thước lớn có hạt nhân." },
        "PLT": { name: "Platelet", full_name: "Tiểu cầu (Platelet / Thrombocyte)", color: "#FD79A8", description: "Mảnh tế bào không nhân có vai trò đông cầm máu." },
        "PMY": { name: "Promyelocyte", full_name: "Tiền tủy bào (Promyelocyte)", color: "#FAB1A0", description: "Giai đoạn kế tiếp nguyên tủy bào, xuất hiện hạt tiên phát." },
        "SNE": { name: "Segmented Neutrophil", full_name: "Bạch cầu trung tính phân đoạn (SNE)", color: "#74B9FF", description: "Bạch cầu trưởng thành phổ biến nhất, nhân chia 2-5 múi." }
    };

    // Global State
    let selectedCellType = null;
    let yoloFile = null;
    let clsFile = null;
    let xaiFile = null;
    let pipeFile = null;
    let cachedXaiResults = null;
    let selectedCropIndices = new Set();  // indices of selected crops
    let allCropData = [];  // all crop data from detection
    let clsQueue = [];  // queue of crop images to classify (base64 strings)
    let clsQueueIndex = 0;  // current index in queue
    let qwenModelsById = new Map();

    // Initialize SPA Router
    function initRouter() {
        const menuItems = document.querySelectorAll(".sidebar-menu a");
        const sections = document.querySelectorAll(".content-section");

        function handleRoute() {
            const hash = window.location.hash || "#dashboard";
            const targetId = hash.replace("#", "");

            menuItems.forEach(item => {
                if (item.getAttribute("href") === hash) {
                    item.classList.add("active");
                } else {
                    item.classList.remove("active");
                }
            });

            sections.forEach(section => {
                if (section.id === targetId) {
                    section.classList.add("active");
                } else {
                    section.classList.remove("active");
                }
            });

            if (targetId === "dataset") {
                loadDatasetExplorer();
            } else if (targetId === "detection") {
                loadSampleImages("yolo-samples", "yolo");
            } else if (targetId === "classification") {
                loadSampleImages("cls-samples", "classification");
            } else if (targetId === "xai") {
                loadSampleImages("xai-samples", "xai");
            } else if (targetId === "compare") {
                initCompareSection();
            }
        }

        window.addEventListener("hashchange", handleRoute);
        handleRoute();
    }

    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/health`);
            const data = await res.json();
            const statusIndicator = document.querySelector(".status-indicator");
            if (data.status === "ok") {
                statusIndicator.classList.remove("offline");
                statusIndicator.classList.add("online");
                statusIndicator.querySelector(".status-text").textContent = "Backend: Connected";
            }
        } catch {
            const statusIndicator = document.querySelector(".status-indicator");
            statusIndicator.classList.remove("online");
            statusIndicator.classList.add("offline");
            statusIndicator.querySelector(".status-text").textContent = "Backend: Disconnected";
        }
    }

    // ───────────────────────────────────────────────────────────
    // Load Samples helper
    // ───────────────────────────────────────────────────────────
    async function loadSampleImages(containerId, moduleType) {
        const container = document.getElementById(containerId);
        if (!container) return;

        try {
            const res = await fetch(`${API_BASE}/samples?limit=4`);
            const data = await res.json();
            
            container.innerHTML = "";
            data.samples.forEach(sample => {
                const div = document.createElement("div");
                div.className = "sample-item";
                div.innerHTML = `<img src="data:image/jpeg;base64,${sample.image_base64}" alt="${sample.filename}">`;
                
                div.addEventListener("click", () => {
                    const blob = base64ToBlob(sample.image_base64, "image/jpeg");
                    
                    if (moduleType === "yolo") {
                        yoloFile = blob;
                        document.getElementById("det-no-image").classList.add("hidden");
                        const img = document.getElementById("det-image-orig");
                        img.src = `data:image/jpeg;base64,${sample.image_base64}`;
                        img.classList.remove("hidden");
                        document.getElementById("det-image-result").classList.add("hidden");
                        document.getElementById("det-stats-container").classList.add("hidden");
                        document.getElementById("det-row-bottom").classList.add("hidden");
                        document.getElementById("crop-result-panel").classList.add("hidden");
                    } else if (moduleType === "classification") {
                        clsFile = blob;
                        document.getElementById("cls-no-image").classList.add("hidden");
                        const img = document.getElementById("cls-image-preview");
                        img.src = `data:image/jpeg;base64,${sample.image_base64}`;
                        img.classList.remove("hidden");
                        document.getElementById("cls-results-container").classList.add("hidden");
                        document.getElementById("cls-image-preview").dataset.sampleType = sample.cell_type;
                        document.getElementById("cls-image-preview").dataset.sampleFilename = sample.filename;
                    } else if (moduleType === "xai") {
                        xaiFile = blob;
                        const grid = document.getElementById("xai-grid");
                        grid.innerHTML = `
                            <div class="xai-card" style="grid-column: span 2;">
                                <h4>Ảnh tế bào gốc</h4>
                                <div class="xai-img-wrapper">
                                    <img src="data:image/jpeg;base64,${sample.image_base64}" alt="Original">
                                </div>
                            </div>
                        `;
                        grid.dataset.sampleType = sample.cell_type;
                        grid.dataset.sampleFilename = sample.filename;
                    }
                });
                container.appendChild(div);
            });
        } catch (err) {
            console.error("Error loading sample images:", err);
        }
    }

    function base64ToBlob(base64Data, contentType = '') {
        const sliceSize = 512;
        const byteCharacters = atob(base64Data);
        const byteArrays = [];
        for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
            const slice = byteCharacters.slice(offset, offset + sliceSize);
            const byteNumbers = new Array(slice.length);
            for (let i = 0; i < slice.length; i++) {
                byteNumbers[i] = slice.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            byteArrays.push(byteArray);
        }
        return new Blob(byteArrays, { type: contentType });
    }

    // ───────────────────────────────────────────────────────────
    // YOLO Detection Logic
    // ───────────────────────────────────────────────────────────
    const yoloUpload = document.getElementById("yolo-upload-area");
    const yoloFileIn = document.getElementById("yolo-file-input");
    
    if (yoloUpload) {
        yoloUpload.addEventListener("click", () => yoloFileIn.click());
        yoloFileIn.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                yoloFile = e.target.files[0];
                const reader = new FileReader();
                reader.onload = (evt) => {
                    document.getElementById("det-no-image").classList.add("hidden");
                    const img = document.getElementById("det-image-orig");
                    img.src = evt.target.result;
                    img.classList.remove("hidden");
                    document.getElementById("det-image-result").classList.add("hidden");
                    document.getElementById("det-stats-container").classList.add("hidden");
                    document.getElementById("det-row-bottom").classList.add("hidden");
                    document.getElementById("crop-result-panel").classList.add("hidden");
                };
                reader.readAsDataURL(yoloFile);
            }
        });
    }

    const confSlider = document.getElementById("yolo-conf");
    if (confSlider) {
        confSlider.addEventListener("input", (e) => {
            document.getElementById("yolo-conf-val").textContent = e.target.value;
        });
    }

    // Update selection count and button state
    function updateSelectionUI() {
        const count = selectedCropIndices.size;
        document.getElementById("det-selected-count").textContent = `Đã chọn: ${count}`;
        document.getElementById("btn-send-to-qwen").disabled = count === 0;
    }

    // Navigate to classification tab with selected crops
    function sendSelectedToQwen() {
        if (selectedCropIndices.size === 0) return;
        
        // Build queue of selected crop images
        clsQueue = [];
        const sortedIndices = Array.from(selectedCropIndices).sort((a, b) => a - b);
        for (const idx of sortedIndices) {
            if (allCropData[idx]) {
                clsQueue.push(allCropData[idx].crop_image_base64);
            }
        }
        clsQueueIndex = 0;
        
        // Load first image in classification tab
        if (clsQueue.length > 0) {
            loadClsQueueImage(0);
        }
        
        // Navigate to classification tab
        window.location.hash = "#classification";
    }
    
    // Load a specific image from the queue into classification preview
    function loadClsQueueImage(index) {
        if (index < 0 || index >= clsQueue.length) return;
        clsQueueIndex = index;
        const cropBase64 = clsQueue[index];
        
        const blob = base64ToBlob(cropBase64, "image/jpeg");
        clsFile = blob;
        
        document.getElementById("cls-no-image").classList.add("hidden");
        const previewImg = document.getElementById("cls-image-preview");
        previewImg.src = `data:image/jpeg;base64,${cropBase64}`;
        previewImg.classList.remove("hidden");
        document.getElementById("cls-results-container").classList.add("hidden");
        
        // Clear sample info
        delete previewImg.dataset.sampleType;
        delete previewImg.dataset.sampleFilename;
        
        // Update queue nav UI
        updateClsQueueNav();
    }
    
    // Update queue navigation indicator
    function updateClsQueueNav() {
        const navContainer = document.getElementById("cls-queue-nav");
        if (!navContainer) return;
        
        if (clsQueue.length > 1) {
            navContainer.classList.remove("hidden");
            navContainer.innerHTML = `
                <span style="font-size: 0.8rem; color: var(--text-secondary); margin-right: 8px;">
                    Ảnh ${clsQueueIndex + 1} / ${clsQueue.length}
                </span>
                <button class="btn btn-cls-nav" id="btn-cls-prev" ${clsQueueIndex === 0 ? 'disabled' : ''} style="padding: 4px 10px; font-size: 0.75rem;">
                    <i class="fa-solid fa-chevron-left"></i>
                </button>
                <button class="btn btn-cls-nav" id="btn-cls-next" ${clsQueueIndex >= clsQueue.length - 1 ? 'disabled' : ''} style="padding: 4px 10px; font-size: 0.75rem;">
                    <i class="fa-solid fa-chevron-right"></i>
                </button>
            `;
            
            document.getElementById("btn-cls-prev").addEventListener("click", () => {
                loadClsQueueImage(clsQueueIndex - 1);
            });
            document.getElementById("btn-cls-next").addEventListener("click", () => {
                loadClsQueueImage(clsQueueIndex + 1);
            });
        } else {
            navContainer.classList.add("hidden");
        }
    }

    // Navigate to classification tab with a single crop (legacy, used by click)
    function navigateToClassificationWithCrop(cropBase64) {
        clsQueue = [cropBase64];
        clsQueueIndex = 0;
        loadClsQueueImage(0);
        window.location.hash = "#classification";
    }

    const btnRunDet = document.getElementById("btn-run-det");
    if (btnRunDet) {
        btnRunDet.addEventListener("click", async () => {
            if (!yoloFile) {
                alert("Vui lòng tải ảnh hoặc chọn ảnh mẫu trước!");
                return;
            }

            btnRunDet.disabled = true;
            btnRunDet.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang chạy YOLO26...';

            const formData = new FormData();
            formData.append("file", yoloFile, "image.jpg");
            formData.append("confidence", confSlider.value);
            
            const selectedModel = document.getElementById("yolo-model-select")?.value;
            if (selectedModel) formData.append("yolo_model", selectedModel);

            try {
                const res = await fetch(`${API_BASE}/detect`, {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                
                if (data.success) {
                    // Show annotated image
                    const imgResult = document.getElementById("det-image-result");
                    imgResult.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;
                    imgResult.classList.remove("hidden");
                    document.getElementById("det-image-orig").classList.add("hidden");

                    // Tab control
                    document.querySelectorAll(".viewer-tab").forEach(tab => {
                        if (tab.dataset.tab === "annotated") tab.classList.add("active");
                        else tab.classList.remove("active");
                    });

                    // Stats
                    document.getElementById("det-stats-container").classList.remove("hidden");
                    document.getElementById("count-rbc").textContent = data.summary.RBC || 0;
                    document.getElementById("count-wbc").textContent = data.summary.WBC || 0;
                    document.getElementById("count-plt").textContent = data.summary.Platelets || 0;

                    // Render crops grid with checkboxes
                    const crops = data.crops || [];
                    const grid = document.getElementById("det-crops-grid");
                    grid.innerHTML = "";
                    
                    // Reset selections
                    selectedCropIndices.clear();
                    allCropData = [];
                    updateSelectionUI();

                    if (crops.length > 0) {
                        crops.forEach((crop, index) => {
                            // Store in global data
                            allCropData.push(crop);
                            
                            const item = document.createElement("div");
                            item.className = "det-crop-item";
                            item.dataset.index = index;
                            item.innerHTML = `
                                <div class="det-crop-img-wrapper">
                                    <div class="det-crop-checkbox">
                                        <input type="checkbox" id="crop-chk-${index}" class="crop-checkbox-input">
                                        <label for="crop-chk-${index}" class="crop-checkbox-label"></label>
                                    </div>
                                    <img src="data:image/jpeg;base64,${crop.crop_image_base64}" alt="Cell #${index + 1}" style="width:100%; height:100%; object-fit:cover;">
                                    <div class="det-crop-badge">#${index + 1}</div>
                                </div>
                                <div class="det-crop-info">
                                    <span class="det-crop-label">${crop.detect_label}</span>
                                    <span class="det-crop-score">${(crop.detect_score * 100).toFixed(1)}%</span>
                                </div>
                            `;
                            
                            // Checkbox toggle selection
                            const checkbox = item.querySelector('.crop-checkbox-input');
                            checkbox.addEventListener('change', (e) => {
                                if (e.target.checked) {
                                    selectedCropIndices.add(index);
                                    item.classList.add('selected');
                                } else {
                                    selectedCropIndices.delete(index);
                                    item.classList.remove('selected');
                                }
                                updateSelectionUI();
                            });
                            
                            // Click on image area → single select (unselect others) + navigate
                            item.querySelector('.det-crop-img-wrapper').addEventListener('click', (e) => {
                                if (e.target.type === 'checkbox') return; // let checkbox handle it
                                
                                // Single-select this crop only
                                selectedCropIndices.clear();
                                selectedCropIndices.add(index);
                                
                                // Update all checkboxes
                                document.querySelectorAll('.crop-checkbox-input').forEach((cb, i) => {
                                    cb.checked = i === index;
                                    cb.closest('.det-crop-item').classList.toggle('selected', i === index);
                                });
                                updateSelectionUI();
                                
                                navigateToClassificationWithCrop(crop.crop_image_base64);
                            });
                            
                            grid.appendChild(item);
                        });
                    } else {
                        grid.innerHTML = '<p class="text-secondary" style="grid-column: 1/-1; text-align: center; padding: 20px;">Không phát hiện tế bào nào</p>';
                    }

                    // Show bottom section
                    document.getElementById("det-row-bottom").classList.remove("hidden");
                    document.getElementById("crop-result-panel").classList.add("hidden");
                } else {
                    alert("Có lỗi khi chạy YOLO26: " + data.error);
                }
            } catch (err) {
                alert("Không thể kết nối đến API server.");
            } finally {
                btnRunDet.disabled = false;
                btnRunDet.innerHTML = '<i class="fa-solid fa-play"></i> Thực hiện phát hiện';
            }
        });
    }

    // Wire "Gửi đến QWen" button
    const btnSendToQwen = document.getElementById("btn-send-to-qwen");
    if (btnSendToQwen) {
        btnSendToQwen.addEventListener("click", sendSelectedToQwen);
    }

    // Tab toggle
    document.querySelectorAll(".viewer-tab").forEach(tab => {
        tab.addEventListener("click", (e) => {
            const type = e.target.dataset.tab;
            document.querySelectorAll(".viewer-tab").forEach(t => t.classList.remove("active"));
            e.target.classList.add("active");

            if (type === "annotated") {
                document.getElementById("det-image-result").classList.remove("hidden");
                document.getElementById("det-image-orig").classList.add("hidden");
            } else {
                document.getElementById("det-image-orig").classList.remove("hidden");
                document.getElementById("det-image-result").classList.add("hidden");
            }
        });
    });

    // ───────────────────────────────────────────────────────────
    // QWen Classification Logic
    // ───────────────────────────────────────────────────────────
    const clsUpload = document.getElementById("cls-upload-area");
    const clsFileIn = document.getElementById("cls-file-input");

    let selectedQwenDevice = "cpu";

    const clsDeviceBtns = document.querySelectorAll("#qwen-device-toggle .device-btn");
    const clsDeviceStatusBadge = document.getElementById("device-status-badge");
    const clsDeviceStatusText  = document.getElementById("device-status-text");

    function updateClsDeviceBadge(device) {
        if (!clsDeviceStatusBadge || !clsDeviceStatusText) return;
        const dot = clsDeviceStatusBadge.querySelector("span:first-child");
        if (dot) {
            dot.className = device === "cuda" ? "dot-cuda" : "dot-cpu";
        }
        if (device === "cuda") {
            clsDeviceStatusText.textContent = "Đang dùng: GPU (CUDA)";
            clsDeviceStatusText.style.color = "#76b900";
        } else {
            clsDeviceStatusText.textContent = "Đang dùng: CPU";
            clsDeviceStatusText.style.color = "";
        }
    }

    clsDeviceBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            clsDeviceBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedQwenDevice = btn.dataset.device;
            updateClsDeviceBadge(selectedQwenDevice);
            const compressionSelect = document.getElementById("qwen-compression-select");
            if (compressionSelect) compressionSelect.value = selectedQwenDevice === "cuda" ? "4bit" : "full";
        });
    });
    updateClsDeviceBadge("cpu");

    if (clsUpload) {
        clsUpload.addEventListener("click", () => clsFileIn.click());
        clsFileIn.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                clsFile = e.target.files[0];
                const reader = new FileReader();
                reader.onload = (evt) => {
                    document.getElementById("cls-no-image").classList.add("hidden");
                    const img = document.getElementById("cls-image-preview");
                    img.src = evt.target.result;
                    img.classList.remove("hidden");
                    document.getElementById("cls-results-container").classList.add("hidden");
                    delete img.dataset.sampleType;
                    delete img.dataset.sampleFilename;
                };
                reader.readAsDataURL(clsFile);
            }
        });
    }

    function addClsLog(message, type = "info") {
        const logBox = document.getElementById("cls-log-box");
        if (!logBox) return;
        const entry = document.createElement("div");
        entry.className = "cls-log-entry";
        const time = new Date().toLocaleTimeString('vi-VN', { hour12: false });
        const timeSpan = document.createElement("span");
        timeSpan.className = "cls-log-time";
        timeSpan.textContent = `[${time}]`;
        const msgSpan = document.createElement("span");
        msgSpan.className = `cls-log-message cls-log-${type}`;
        msgSpan.textContent = message;
        entry.appendChild(timeSpan);
        entry.appendChild(msgSpan);
        logBox.appendChild(entry);
        logBox.scrollTop = logBox.scrollHeight;
    }

    const btnRunCls = document.getElementById("btn-run-cls");
    if (btnRunCls) {
        btnRunCls.addEventListener("click", async () => {
            if (!clsFile) {
                alert("Vui lòng tải ảnh tế bào hoặc chọn mẫu!");
                return;
            }

            btnRunCls.disabled = true;
            btnRunCls.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> QWen đang suy luận...';

            const logContainer = document.getElementById("cls-log-container");
            const logBox = document.getElementById("cls-log-box");
            if (logContainer && logBox) {
                logBox.innerHTML = "";
                logContainer.classList.remove("hidden");
                addClsLog("Bắt đầu phân tích ảnh tế bào...", "info");
            }

            const previewImg = document.getElementById("cls-image-preview");
            const sampleType = previewImg.dataset.sampleType;
            const sampleFilename = previewImg.dataset.sampleFilename;

            let url = `${API_BASE}/classify`;
            const formData = new FormData();

            if (sampleType && sampleFilename) {
                url = `${API_BASE}/classify-sample`;
                formData.append("cell_type", sampleType);
                formData.append("filename", sampleFilename);
                addClsLog(`Sử dụng mẫu: ${sampleType}/${sampleFilename}`, "info");
            } else {
                formData.append("file", clsFile, "cell.jpg");
                addClsLog("Tải lên ảnh tế bào tùy chỉnh", "info");
            }
            
            const selectedQwenModel = document.getElementById("qwen-model-select")?.value;
            if (selectedQwenModel) {
                formData.append("qwen_model", selectedQwenModel);
                addClsLog(`Model được chọn: ${qwenModelsById.get(selectedQwenModel)?.name || selectedQwenModel}`, "info");
            }

            formData.append("device", selectedQwenDevice);
            
            const selectedQwenCompression = document.getElementById("qwen-compression-select")?.value;
            if (selectedQwenCompression) {
                formData.append("qwen_compression", selectedQwenCompression);
            }

            addClsLog(`Thiết bị: ${selectedQwenDevice.toUpperCase()}`, selectedQwenDevice === "cuda" ? "success" : "info");

            try {
                addClsLog("Đang gửi yêu cầu đến server...", "info");
                const res = await fetch(url, { method: "POST", body: formData });
                const data = await res.json();

                if (data.success) {
                    document.getElementById("cls-results-container").classList.remove("hidden");

                    const cellInfo = CELL_TYPES_DATA[data.predicted_class] || { name: data.predicted_class || "—", full_name: "Tế bào", color: "#00b894", description: "" };

                    const badge = document.getElementById("cls-badge");
                    badge.textContent = data.predicted_class || "—";
                    badge.style.backgroundColor = cellInfo.color || "#00b894";

                    document.getElementById("cls-name").textContent = `${data.predicted_class || "—"} (${cellInfo.name || ""})`;
                    const descEl = document.getElementById("cls-full-desc");
                    if (descEl) descEl.textContent = cellInfo.description || "";

                    const confPercent = data.confidence_percent || (data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : "—%");
                    const confVal = data.confidence ? (data.confidence * 100) : 0;

                    const confBadge = document.getElementById("cls-conf-badge");
                    if (confBadge) confBadge.textContent = confPercent;

                    const confText = document.getElementById("cls-conf-text");
                    if (confText) confText.textContent = confPercent;

                    const confBar = document.getElementById("cls-conf-bar");
                    if (confBar) confBar.style.width = `${Math.min(100, Math.max(0, confVal))}%`;

                    // Render Top Probabilities
                    const probsList = document.getElementById("cls-probs-list");
                    if (probsList && data.top_probabilities && data.top_probabilities.length > 0) {
                        probsList.innerHTML = "";
                        data.top_probabilities.slice(0, 3).forEach((item, idx) => {
                            const pPercent = item.percentage || `${(item.probability * 100).toFixed(1)}%`;
                            const pWidth = Math.min(100, Math.max(0, item.probability * 100));
                            const pColor = item.color || CELL_TYPES_DATA[item.class]?.color || "#00cec9";

                            const row = document.createElement("div");
                            row.style.display = "flex";
                            row.style.alignItems = "center";
                            row.style.gap = "8px";
                            row.style.fontSize = "0.8rem";
                            row.innerHTML = `
                                <span style="font-weight: 700; width: 32px; color: ${pColor};">${item.class}</span>
                                <span style="flex: 1; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${item.name || item.full_name}</span>
                                <div style="flex: 1.5; height: 6px; background: var(--bg-tertiary, #2d3436); border-radius: 3px; overflow: hidden;">
                                    <div style="height: 100%; width: ${pWidth}%; background: ${pColor}; border-radius: 3px; transition: width 0.3s ease;"></div>
                                </div>
                                <span style="font-weight: 600; width: 44px; text-align: right; color: var(--text-primary);">${pPercent}</span>
                            `;
                            probsList.appendChild(row);
                        });
                    }

                    document.getElementById("cls-raw-output").textContent = data.raw_output || "(trống)";
                    document.getElementById("cls-model-used-text").textContent = data.model_used || "—";
                    document.getElementById("cls-inference-time").textContent = `${data.inference_time_ms ?? "—"} ms`;

                    if (data.model_used) {
                        addClsLog(`Mô hình thực thi: ${data.model_used}`, "info");
                    }
                    addClsLog(`Dự đoán: ${data.predicted_class || "Không nhận diện được"} (${confPercent})`, data.predicted_class ? "success" : "warning");
                    addClsLog(`Raw output: ${data.raw_output}`, "info");
                    addClsLog(`Thời gian inference: ${data.inference_time_ms} ms`, "info");
                    
                    if (data.ground_truth) {
                        const isCorrect = data.is_correct;
                        addClsLog(`Ground truth: ${data.ground_truth} (${data.ground_truth_name})`, 
                                  isCorrect ? "success" : "warning");
                        if (!isCorrect) {
                            addClsLog(`⚠️ Sai lệch! Dự đoán khác với nhãn thực tế`, "error");
                        }
                    }

                    // Record run to Compare History
                    const previewImgSrc = document.getElementById("cls-image-preview")?.src;
                    saveClassificationRun({
                        image_src: previewImgSrc || "",
                        model_id: selectedQwenModel,
                        model_name: data.model_used || qwenModelsById.get(selectedQwenModel)?.name || "QWen Model",
                        predicted_class: data.predicted_class,
                        cell_name: cellInfo.name,
                        color: cellInfo.color,
                        confidence_percent: confPercent,
                        top_probabilities: data.top_probabilities || [],
                        inference_time_ms: data.inference_time_ms,
                        device: selectedQwenDevice
                    });
                } else {
                    const message = data.error || data.detail || "Lỗi inference không xác định";
                    addClsLog(`Lỗi: ${message}`, "error");
                    alert("Phân loại thất bại: " + message);
                }
            } catch (err) {
                addClsLog(`Lỗi kết nối: ${err.message}`, "error");
                alert("Lỗi kết nối đến Backend.");
            } finally {
                btnRunCls.disabled = false;
                btnRunCls.innerHTML = '<i class="fa-solid fa-brain"></i> QWen phân loại tế bào';
            }
        });
    }
    
    if (clsUpload) {
        clsFileIn.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                const logContainer = document.getElementById("cls-log-container");
                if (logContainer) logContainer.classList.add("hidden");
            }
        });
    }
    
    const clsSamples = document.getElementById("cls-samples");
    if (clsSamples) {
        clsSamples.addEventListener("click", () => {
            const logContainer = document.getElementById("cls-log-container");
            if (logContainer) logContainer.classList.add("hidden");
        });
    }

    // ───────────────────────────────────────────────────────────
    // XAI Engine Logic
    // ───────────────────────────────────────────────────────────
    const xaiUpload = document.getElementById("xai-upload-area");
    const xaiFileIn = document.getElementById("xai-file-input");

    if (xaiUpload) {
        xaiUpload.addEventListener("click", () => xaiFileIn.click());
        xaiFileIn.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                xaiFile = e.target.files[0];
                const reader = new FileReader();
                reader.onload = (evt) => {
                    const grid = document.getElementById("xai-grid");
                    grid.innerHTML = `
                        <div class="xai-card" style="grid-column: span 2;">
                            <h4>Ảnh tế bào gốc</h4>
                            <div class="xai-img-wrapper">
                                <img src="${evt.target.result}" alt="Original">
                            </div>
                        </div>
                    `;
                    delete grid.dataset.sampleType;
                    delete grid.dataset.sampleFilename;
                };
                reader.readAsDataURL(xaiFile);
            }
        });
    }

    const alphaSlider = document.getElementById("xai-alpha");
    if (alphaSlider) {
        alphaSlider.addEventListener("input", (e) => {
            document.getElementById("xai-alpha-val").textContent = e.target.value;
        });
    }

    const btnRunXai = document.getElementById("btn-run-xai");
    const btnRunXaiAll = document.getElementById("btn-run-xai-all");

    if (btnRunXai) btnRunXai.addEventListener("click", () => renderXai(false));
    if (btnRunXaiAll) btnRunXaiAll.addEventListener("click", () => renderXai(true));

    async function renderXai(allMethods = false) {
        if (!xaiFile) {
            alert("Vui lòng tải hoặc chọn ảnh mẫu để tạo giải thích XAI!");
            return;
        }

        const btn = allMethods ? btnRunXaiAll : btnRunXai;
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang tính toán Heatmap...';

        const grid = document.getElementById("xai-grid");
        const sampleType = grid.dataset.sampleType;
        const sampleFilename = grid.dataset.sampleFilename;
        const method = document.getElementById("pipe-xai-method")?.value || "HiResCAM";

        let url = allMethods ? `${API_BASE}/xai-all` : `${API_BASE}/xai`;
        const formData = new FormData();
        formData.append("alpha", alphaSlider.value);
        formData.append("file", xaiFile, "cell.jpg");
        if (!allMethods) formData.append("method", method);

        try {
            const res = await fetch(url, { method: "POST", body: formData });
            const data = await res.json();

            if (data.success) {
                grid.innerHTML = "";
                if (allMethods) {
                    const methods = ["HiResCAM", "XGradCAM", "EigenCAM", "IntegratedGradients"];
                    methods.forEach(m => {
                        const result = data.results[m];
                        const card = document.createElement("div");
                        card.className = "xai-card";
                        card.innerHTML = `
                            <h4><span class="xai-card-badge" style="background-color: ${result.method_info.color};"></span>${result.method_info.name}</h4>
                            <div class="xai-img-wrapper"><img src="data:image/jpeg;base64,${result.overlay_base64}" alt="${m}"></div>
                            <p>${result.method_info.description}</p>
                        `;
                        grid.appendChild(card);
                    });
                } else {
                    grid.innerHTML = `
                        <div class="xai-card">
                            <h4>Ảnh tế bào gốc</h4>
                            <div class="xai-img-wrapper"><img src="data:image/jpeg;base64,${data.original_base64}" alt="Orig"></div>
                        </div>
                        <div class="xai-card">
                            <h4><span class="xai-card-badge" style="background-color: ${data.method_info.color};"></span>${data.method_info.name} Heatmap</h4>
                            <div class="xai-img-wrapper"><img src="data:image/jpeg;base64,${data.overlay_base64}" alt="XAI"></div>
                            <p>${data.method_info.description}</p>
                        </div>
                    `;
                }
            } else {
                alert("Có lỗi khi tạo XAI: " + data.error);
            }
        } catch (err) {
            alert("Lỗi kết nối Backend.");
        } finally {
            btn.disabled = false;
            btnRunXai.innerHTML = '<i class="fa-solid fa-bolt"></i> Tạo Heatmap';
            btnRunXaiAll.innerHTML = '<i class="fa-solid fa-grid-2"></i> So sánh 4 Methods';
        }
    }

    // ───────────────────────────────────────────────────────────
    // Pipeline Integration Logic
    // ───────────────────────────────────────────────────────────
    const pipeUpload = document.getElementById("pipe-upload-area");
    const pipeFileIn = document.getElementById("pipe-file-input");

    if (pipeUpload) {
        pipeUpload.addEventListener("click", () => pipeFileIn.click());
        pipeFileIn.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                pipeFile = e.target.files[0];
                document.getElementById("pipe-empty").innerHTML = `
                    <i class="fa-solid fa-check text-success"></i>
                    <span>Tiêu bản đã được tải lên thành công. Nhấn nút để khởi chạy Pipeline.</span>
                `;
            }
        });
    }

    const pipeConf = document.getElementById("pipe-conf");
    if (pipeConf) {
        pipeConf.addEventListener("input", (e) => {
            document.getElementById("pipe-conf-val").textContent = e.target.value;
        });
    }

    let selectedPipeDevice = "cpu";
    const pipeDeviceBtns = document.querySelectorAll("#pipe-device-toggle .device-btn");
    const pipeDeviceStatusText = document.getElementById("pipe-device-status-text");
    const pipeDeviceStatusBadge = document.getElementById("pipe-device-status-badge");

    function updatePipeDeviceBadge(device) {
        if (!pipeDeviceStatusText || !pipeDeviceStatusBadge) return;
        const dot = pipeDeviceStatusBadge.querySelector("span:first-child");
        if (dot) dot.className = device === "cuda" ? "dot-cuda" : "dot-cpu";
        if (device === "cuda") {
            pipeDeviceStatusText.textContent = "Đang dùng: GPU (CUDA)";
            pipeDeviceStatusText.style.color = "#76b900";
        } else {
            pipeDeviceStatusText.textContent = "Đang dùng: CPU";
            pipeDeviceStatusText.style.color = "";
        }
    }

    pipeDeviceBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            pipeDeviceBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedPipeDevice = btn.dataset.device;
            updatePipeDeviceBadge(selectedPipeDevice);
            const compressionSelect = document.getElementById("pipe-qwen-compression");
            if (compressionSelect) compressionSelect.value = selectedPipeDevice === "cuda" ? "4bit" : "full";
        });
    });

    const btnRunPipe = document.getElementById("btn-run-pipeline");
    if (btnRunPipe) {
        btnRunPipe.addEventListener("click", async () => {
            if (!pipeFile) {
                alert("Vui lòng tải tệp ảnh tiêu bản blood smear thô lên!");
                return;
            }

            btnRunPipe.disabled = true;
            btnRunPipe.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang chạy Pipeline...';

            const formData = new FormData();
            formData.append("file", pipeFile, "smear.jpg");
            formData.append("confidence", pipeConf.value);
            formData.append("xai_method", document.getElementById("pipe-xai-method").value);
            formData.append("xai_alpha", 0.5);
            formData.append("device", selectedPipeDevice);
            
            const selectedPipeQwenCompression = document.getElementById("pipe-qwen-compression")?.value;
            if (selectedPipeQwenCompression) {
                formData.append("qwen_compression", selectedPipeQwenCompression);
            }
            
            const selectedPipeYolo = document.getElementById("pipe-yolo-model")?.value;
            if (selectedPipeYolo) formData.append("yolo_model", selectedPipeYolo);
            
            const selectedPipeQwen = document.getElementById("pipe-qwen-model")?.value;
            if (selectedPipeQwen) formData.append("qwen_model", selectedPipeQwen);

            try {
                const res = await fetch(`${API_BASE}/pipeline`, { method: "POST", body: formData });
                const data = await res.json();

                if (data.success) {
                    document.getElementById("pipe-empty").classList.add("hidden");
                    document.getElementById("pipe-results").classList.remove("hidden");

                    document.getElementById("pipe-det-img").src = `data:image/jpeg;base64,${data.detection.annotated_image_base64}`;
                    
                    const summaryList = document.getElementById("pipe-summary-list");
                    summaryList.innerHTML = "";
                    for (const [cls, count] of Object.entries(data.detection.summary)) {
                        const li = document.createElement("li");
                        li.innerHTML = `<strong>${cls}:</strong> ${count} tế bào phát hiện`;
                        summaryList.appendChild(li);
                    }

                    const cropsList = document.getElementById("pipe-crops-list");
                    cropsList.innerHTML = "";
                    data.classifications.forEach(cell => {
                        const item = document.createElement("div");
                        item.className = "crop-pipe-item";
                        item.innerHTML = `
                            <img class="crop-pipe-img" src="data:image/jpeg;base64,${cell.crop_image_base64}" alt="crop">
                            <div class="crop-pipe-meta">
                                <h5>YOLO Detect: ${cell.detect_label}</h5>
                                <p>Độ tự tin detect: ${(cell.detect_score * 100).toFixed(1)}%</p>
                                <p>Box: [${cell.box.join(", ")}]</p>
                            </div>
                            <div class="crop-pipe-meta">
                                <h5>QWen Class: <strong>${cell.classification.predicted_class || "Không xác định"}</strong></h5>
                                <p>Raw output: ${cell.classification.raw_output || "(trống)"}</p>
                                <p>Model: ${cell.classification.model_used}</p>
                                <p>Inference: ${cell.classification.inference_time_ms} ms</p>
                            </div>
                            <div>
                                <h5>XAI Explanation</h5>
                                <img class="crop-pipe-img" src="data:image/jpeg;base64,${cell.xai.overlay_base64}" alt="xai">
                            </div>
                        `;
                        cropsList.appendChild(item);
                    });
                } else {
                    alert("Lỗi Pipeline: " + data.error);
                }
            } catch (err) {
                alert("Lỗi kết nối Backend.");
            } finally {
                btnRunPipe.disabled = false;
                btnRunPipe.innerHTML = '<i class="fa-solid fa-gears"></i> Chạy Pipeline';
            }
        });
    }

    // ───────────────────────────────────────────────────────────
    // Dataset Explorer Logic
    // ───────────────────────────────────────────────────────────
    async function loadDatasetExplorer() {
        const listContainer = document.getElementById("dataset-types-list");
        if (!listContainer) return;

        try {
            const res = await fetch(`${API_BASE}/cell-types`);
            const data = await res.json();

            listContainer.innerHTML = "";
            data.cell_types.forEach(cell => {
                const item = document.createElement("div");
                item.className = "cell-type-item";
                item.dataset.code = cell.code;
                item.innerHTML = `
                    <div class="cell-type-left">
                        <div class="cell-type-color" style="background-color: ${cell.color};"></div>
                        <strong>${cell.code}</strong>
                        <span>- ${cell.name}</span>
                    </div>
                    <span class="cell-count-badge">${cell.sample_count} ảnh</span>
                `;
                item.addEventListener("click", () => {
                    document.querySelectorAll(".cell-type-item").forEach(i => i.classList.remove("active"));
                    item.classList.add("active");
                    showCellDetail(cell);
                });
                listContainer.appendChild(item);
            });
        } catch (err) {
            console.error("Error loading dataset explorer:", err);
        }
    }

    async function showCellDetail(cell) {
        document.getElementById("dataset-no-select").classList.add("hidden");
        const detail = document.getElementById("dataset-detail");
        detail.classList.remove("hidden");

        const badge = document.getElementById("detail-code");
        badge.textContent = cell.code;
        badge.style.backgroundColor = cell.color;

        document.getElementById("detail-name").textContent = cell.full_name;
        document.getElementById("detail-desc").textContent = cell.description;

        const samplesGrid = document.getElementById("detail-samples");
        samplesGrid.innerHTML = '<i class="fa-solid fa-spinner fa-spin col-span-4 text-center"></i>';

        try {
            const res = await fetch(`${API_BASE}/samples?cell_type=${cell.code}&limit=8`);
            const data = await res.json();
            samplesGrid.innerHTML = "";
            data.samples.forEach(sample => {
                const imgDiv = document.createElement("div");
                imgDiv.className = "detail-sample-img";
                imgDiv.innerHTML = `<img src="data:image/jpeg;base64,${sample.image_base64}" alt="${cell.code}">`;
                samplesGrid.appendChild(imgDiv);
            });
        } catch (err) {
            samplesGrid.innerHTML = '<p class="text-danger">Không tải được ảnh mẫu.</p>';
        }
    }

    // ───────────────────────────────────────────────────────────
    // Load Available Models Logic
    // ───────────────────────────────────────────────────────────
    async function loadModels() {
        try {
            const res = await fetch(`${API_BASE}/models`);
            const data = await res.json();
            
            const populateSelect = (selectId, models) => {
                const select = document.getElementById(selectId);
                if (select) {
                    select.innerHTML = "";
                    models.forEach(m => {
                        const opt = document.createElement("option");
                        opt.value = m.id;
                        opt.textContent = m.name;
                        opt.disabled = m.available === false;
                        if (m.error) opt.title = m.error;
                        select.appendChild(opt);
                    });
                }
            };
            
            populateSelect("yolo-model-select", data.yolo_models);
            populateSelect("pipe-yolo-model", data.yolo_models);
            qwenModelsById = new Map(data.qwen_models.map(model => [model.id, model]));
            populateSelect("qwen-model-select", data.qwen_models);
            populateSelect("pipe-qwen-model", data.qwen_models);
        } catch (err) {
            console.error("Error loading models:", err);
        }
    }

    // ───────────────────────────────────────────────────────────
    // COMPARE & HISTORY SYSTEM — Lịch sử 3 Mô Hình Gần Nhất
    // ───────────────────────────────────────────────────────────
    function getModelCategory(modelId, modelName) {
        const idLower = (modelId || "").toLowerCase();
        const nameLower = (modelName || "").toLowerCase();
        if (idLower.includes("qlora") || idLower.includes("quoc-huy") || nameLower.includes("quốc huy") || nameLower.includes("qlora")) {
            return "qlora";
        }
        if (idLower.includes("lora-r16") || idLower.includes("5500") || (nameLower.includes("lora") && !nameLower.includes("dora") && !nameLower.includes("qlora"))) {
            return "lora";
        }
        if (idLower.includes("dora") || idLower.includes("3315") || nameLower.includes("dora") || nameLower.includes("nhật hào")) {
            return "dora";
        }
        return "dora"; // default fallback
    }

    function saveClassificationRun(runData) {
        try {
            const timestamp = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
            const dateStr = new Date().toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit" });
            const category = getModelCategory(runData.model_id, runData.model_name);

            const record = {
                ...runData,
                category: category,
                timestamp: `${timestamp} (${dateStr})`
            };

            // Save as latest run for this model category
            localStorage.setItem(`hemoai_latest_${category}`, JSON.stringify(record));

            // Append to full history list
            let history = [];
            try {
                history = JSON.parse(localStorage.getItem("hemoai_classification_history") || "[]");
            } catch (e) { history = []; }

            history.unshift(record);
            if (history.length > 50) history = history.slice(0, 50);
            localStorage.setItem("hemoai_classification_history", JSON.stringify(history));

            // Refresh compare UI
            renderCompareHistoryUI();
        } catch (e) {
            console.error("Error saving classification run:", e);
        }
    }

    function renderCompareHistoryUI() {
        const categories = [
            { key: "qlora", bodyId: "body-model-qlora", title: "Quốc Huy — Qwen2-VL-2B (QLoRA)", color: "#6c5ce7" },
            { key: "lora",  bodyId: "body-model-lora",  title: "LoRA Ckpt-5500 (3B)", color: "#0984e3" },
            { key: "dora",  bodyId: "body-model-dora",  title: "Nhật Hào — DoRA Ckpt-3315 (3B)", color: "#00b894" }
        ];

        // 1. Render 3 model cards
        categories.forEach(cat => {
            const bodyEl = document.getElementById(cat.bodyId);
            if (!bodyEl) return;

            let data = null;
            try {
                const stored = localStorage.getItem(`hemoai_latest_${cat.key}`);
                if (stored) data = JSON.parse(stored);
            } catch (e) {}

            if (!data) {
                bodyEl.innerHTML = `
                    <div class="empty-state-card" style="text-align: center; padding: 24px 12px; color: var(--text-secondary);">
                        <i class="fa-solid fa-clock-rotate-left" style="font-size: 1.8rem; margin-bottom: 8px; opacity: 0.5;"></i>
                        <p style="font-size: 0.85rem; margin: 0;">Chưa có lượt chạy gần nhất.<br>Hãy chọn model này tại tab <strong>Phân loại</strong> để xem kết quả đối chiếu.</p>
                    </div>
                `;
                return;
            }

            const cellInfo = CELL_TYPES_DATA[data.predicted_class] || { name: data.predicted_class || "—", color: cat.color };
            const confVal = data.confidence_percent || "—%";
            const topProbs = data.top_probabilities || [];

            let probsHtml = "";
            if (topProbs.length > 0) {
                probsHtml = `
                    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border-color); font-size: 0.78rem;">
                        <div style="font-weight: 600; color: var(--text-secondary); margin-bottom: 4px;">Top xác suất (Softmax):</div>
                        <div style="display: flex; flex-direction: column; gap: 4px;">
                            ${topProbs.slice(0, 3).map(p => {
                                const pWidth = Math.min(100, Math.max(0, (p.probability || 0) * 100));
                                return `
                                    <div style="display: flex; align-items: center; gap: 6px;">
                                        <span style="font-weight: 700; width: 28px; color: ${p.color || '#00b894'};">${p.class}</span>
                                        <div style="flex: 1; height: 5px; background: var(--bg-tertiary, #2d3436); border-radius: 3px; overflow: hidden;">
                                            <div style="height: 100%; width: ${pWidth}%; background: ${p.color || cat.color};"></div>
                                        </div>
                                        <span style="font-weight: 600; width: 38px; text-align: right; color: var(--text-primary);">${p.percentage || (p.probability * 100).toFixed(1) + '%'}</span>
                                    </div>
                                `;
                            }).join("")}
                        </div>
                    </div>
                `;
            }

            bodyEl.innerHTML = `
                <div style="display: flex; gap: 12px; align-items: center;">
                    ${data.image_src ? `<img src="${data.image_src}" alt="Cell" style="width: 64px; height: 64px; object-fit: cover; border-radius: 8px; border: 1px solid var(--border-color);">` : ''}
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="result-badge" style="background-color: ${cellInfo.color}; font-size: 0.85rem; padding: 2px 8px; border-radius: 6px;">${data.predicted_class || '—'}</span>
                            <span class="badge" style="background: rgba(0, 184, 148, 0.15); color: #00b894; font-weight: 700; font-size: 0.85rem;">${confVal}</span>
                        </div>
                        <h5 style="margin: 4px 0 2px 0; font-size: 0.95rem;">${cellInfo.name || data.predicted_class || '—'}</h5>
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-secondary);">
                            <span><i class="fa-solid fa-gauge-high"></i> ${data.inference_time_ms ? data.inference_time_ms + ' ms' : '—'}</span>
                            <span><i class="fa-regular fa-clock"></i> ${data.timestamp || 'Vừa xong'}</span>
                        </div>
                    </div>
                </div>
                ${probsHtml}
            `;
        });

        // 2. Render Full History Table
        const tbody = document.getElementById("compare-history-tbody");
        if (!tbody) return;

        let history = [];
        try {
            history = JSON.parse(localStorage.getItem("hemoai_classification_history") || "[]");
        } catch (e) {}

        if (history.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" style="text-align: center; padding: 24px; color: var(--text-secondary);">Chưa có dữ liệu lịch sử phân loại. Hãy chạy thử một vài ảnh tại tab Phân loại.</td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = history.map((item, idx) => {
            const cellInfo = CELL_TYPES_DATA[item.predicted_class] || { name: item.predicted_class || "—", color: "#00b894" };
            const methodTag = item.category === "qlora" ? '<span class="badge" style="background: rgba(108, 92, 231, 0.15); color: #6c5ce7;">QLoRA (2B)</span>'
                            : item.category === "lora"  ? '<span class="badge" style="background: rgba(9, 132, 227, 0.15); color: #0984e3;">LoRA (3B)</span>'
                            : '<span class="badge" style="background: rgba(0, 184, 148, 0.15); color: #00b894;">DoRA (3B)</span>';

            return `
                <tr style="border-bottom: 1px solid var(--border-color);">
                    <td style="padding: 8px 12px; font-size: 0.8rem; color: var(--text-secondary);">${item.timestamp || '—'}</td>
                    <td style="padding: 8px 12px;">
                        ${item.image_src ? `<img src="${item.image_src}" alt="Crop" style="width: 36px; height: 36px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border-color);">` : '—'}
                    </td>
                    <td style="padding: 8px 12px; font-weight: 500;">${item.model_name || 'QWen'}</td>
                    <td style="padding: 8px 12px;">${methodTag}</td>
                    <td style="padding: 8px 12px;">
                        <span class="result-badge" style="background-color: ${cellInfo.color}; font-size: 0.75rem; padding: 2px 6px; border-radius: 4px;">${item.predicted_class || '—'}</span>
                        <span style="margin-left: 4px; font-size: 0.8rem; color: var(--text-secondary);">${cellInfo.name || ''}</span>
                    </td>
                    <td style="padding: 8px 12px; font-weight: 700; color: #00b894;">${item.confidence_percent || '—'}</td>
                    <td style="padding: 8px 12px; color: var(--text-secondary);">${item.inference_time_ms ? item.inference_time_ms + ' ms' : '—'}</td>
                </tr>
            `;
        }).join("");
    }

    function initCompareSection() {
        const btnClear = document.getElementById("btn-clear-compare-history");
        if (btnClear) {
            btnClear.addEventListener("click", () => {
                if (confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử chạy của các mô hình không?")) {
                    localStorage.removeItem("hemoai_latest_qlora");
                    localStorage.removeItem("hemoai_latest_lora");
                    localStorage.removeItem("hemoai_latest_dora");
                    localStorage.removeItem("hemoai_classification_history");
                    renderCompareHistoryUI();
                }
            });
        }
        renderCompareHistoryUI();
    }

    // Startup
    initRouter();
    checkHealth();
    loadModels();
    initCompareSection();
    setInterval(checkHealth, 10000);
});
