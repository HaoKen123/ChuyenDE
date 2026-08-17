/* Single Page Routing & Frontend Logic */

document.addEventListener("DOMContentLoaded", () => {
    // API config
    const API_BASE = "http://localhost:8000/api";
    
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
                addClsLog(`Model được chọn: ${selectedQwenModel.split(/[\\\/]/).pop()}`, "info");
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
                    
                    const badge = document.getElementById("cls-badge");
                    badge.textContent = data.predicted_class;
                    badge.style.backgroundColor = data.color;

                    document.getElementById("cls-name").textContent = data.class_name;
                    document.getElementById("cls-confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
                    
                    const modelUsedSpan = document.getElementById("cls-model-used-text");
                    if (modelUsedSpan && data.model_used) {
                        modelUsedSpan.textContent = `| Model: ${data.model_used}`;
                    }

                    document.getElementById("cls-desc").textContent = data.description;

                    const probList = document.getElementById("cls-prob-list");
                    probList.innerHTML = "";
                    data.top_predictions.forEach(pred => {
                        const percent = (pred.confidence * 100).toFixed(1);
                        const progress = document.createElement("div");
                        progress.className = "progress-bar-container";
                        progress.innerHTML = `
                            <div class="progress-label">
                                <span>${pred.name} (${pred.class})</span>
                                <strong>${percent}%</strong>
                            </div>
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" style="width: ${percent}%; background-color: ${pred.color};"></div>
                            </div>
                        `;
                        probList.appendChild(progress);
                    });

                    if (data.model_used) {
                        addClsLog(`Mô hình thực thi: ${data.model_used}`, "info");
                    }
                    addClsLog(`Dự đoán: ${data.predicted_class} (${data.class_name})`, "success");
                    addClsLog(`Độ tin cậy: ${(data.confidence * 100).toFixed(1)}%`, "success");
                    addClsLog(`Raw output: ${data.raw_output}`, "info");
                    
                    if (data.ground_truth) {
                        const isCorrect = data.is_correct;
                        addClsLog(`Ground truth: ${data.ground_truth} (${data.ground_truth_name})`, 
                                  isCorrect ? "success" : "warning");
                        if (!isCorrect) {
                            addClsLog(`⚠️ Sai lệch! Dự đoán khác với nhãn thực tế`, "error");
                        }
                    }
                } else {
                    addClsLog(`Lỗi: ${data.error}`, "error");
                    alert("Phân loại thất bại: " + data.error);
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
                        const percent = (cell.classification.confidence * 100).toFixed(1);
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
                                <h5>QWen Class: <span style="color: ${cell.classification.color}; font-weight:800;">${cell.classification.predicted_class}</span></h5>
                                <p>${cell.classification.class_name}</p>
                                <p>Độ tin cậy: ${percent}%</p>
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
                        select.appendChild(opt);
                    });
                }
            };
            
            populateSelect("yolo-model-select", data.yolo_models);
            populateSelect("pipe-yolo-model", data.yolo_models);
            populateSelect("qwen-model-select", data.qwen_models);
            populateSelect("pipe-qwen-model", data.qwen_models);
            // Compare page selects
            populateSelect("compare-model-a-select", data.qwen_models);
            populateSelect("compare-model-b-select", data.qwen_models);
            // Auto-select: Model A = checkpoint-5500 (index 0 = best fine-tuned), Model B = base or second
            const selA = document.getElementById("compare-model-a-select");
            const selB = document.getElementById("compare-model-b-select");
            if (selA && data.qwen_models.length > 0) selA.value = data.qwen_models[0].id;
            if (selB && data.qwen_models.length > 1) selB.value = data.qwen_models[1].id;
            else if (selB && data.qwen_models.length > 0) selB.value = data.qwen_models[0].id;
        } catch (err) {
            console.error("Error loading models:", err);
        }
    }

    // ───────────────────────────────────────────────────────────
    // COMPARE SECTION — So sánh 2 Models
    // ───────────────────────────────────────────────────────────
    let compareFile = null;
    let compareDevice = "cpu";
    let compareInited = false;

    function initCompareSection() {
        if (compareInited) return;  // Chỉ init 1 lần
        compareInited = true;

        const dropZone   = document.getElementById("compare-drop-zone");
        const fileInput  = document.getElementById("compare-file-input");
        const previewWr  = document.getElementById("compare-preview-wrapper");
        const previewImg = document.getElementById("compare-preview-img");
        const removeBtn  = document.getElementById("compare-remove-img");
        const runBtn     = document.getElementById("compare-run-btn");
        const chipsEl    = document.getElementById("compare-sample-chips");

        // ── Device toggle ────────────────────────────────────────
        ["compare-device-cpu", "compare-device-gpu"].forEach(id => {
            const btn = document.getElementById(id);
            if (!btn) return;
            btn.addEventListener("click", () => {
                compareDevice = btn.dataset.device;
                document.querySelectorAll("#compare section .device-btn, .compare-action-row .device-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
            });
        });

        // ── Sample chips ─────────────────────────────────────────
        const SAMPLE_LABELS = ["LY", "SNE", "MO", "EO", "BA", "PLT", "ERB", "BNE", "MMY", "MY", "MYO", "PMY"];
        SAMPLE_LABELS.forEach(lbl => {
            const chip = document.createElement("button");
            chip.className = "compare-sample-chip";
            chip.textContent = lbl;
            chip.title = `Tải mẫu ${lbl} từ dataset`;
            chip.addEventListener("click", async () => {
                try {
                    chip.disabled = true;
                    chip.textContent = "...";
                    // Lấy 1 ảnh mẫu ngẫu nhiên từ API
                    const res = await fetch(`${API_BASE}/samples?cell_type=${lbl}&limit=1`);
                    const data = await res.json();
                    if (data.samples && data.samples.length > 0) {
                        const s = data.samples[0];
                        const imgUrl = `data:image/jpeg;base64,${s.image_base64}`;
                        // Convert base64 to File object
                        const blob = await (await fetch(imgUrl)).blob();
                        compareFile = new File([blob], `${lbl}_sample.jpg`, { type: "image/jpeg" });
                        previewImg.src = imgUrl;
                        previewWr.style.display = "block";
                        dropZone.style.display = "none";
                        runBtn.disabled = false;
                        // Highlight active chip
                        document.querySelectorAll(".compare-sample-chip").forEach(c => c.classList.remove("active"));
                        chip.classList.add("active");
                    }
                } catch(e) {
                    console.error("Error loading sample:", e);
                } finally {
                    chip.disabled = false;
                    chip.textContent = lbl;
                }
            });
            chipsEl.appendChild(chip);
        });

        // ── Dropzone upload ───────────────────────────────────────
        dropZone.addEventListener("click", () => fileInput.click());
        dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
        dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
        dropZone.addEventListener("drop", e => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith("image/")) setCompareFile(file);
        });
        fileInput.addEventListener("change", () => {
            if (fileInput.files[0]) setCompareFile(fileInput.files[0]);
        });
        removeBtn.addEventListener("click", () => {
            compareFile = null;
            previewWr.style.display = "none";
            dropZone.style.display = "";
            runBtn.disabled = true;
            document.querySelectorAll(".compare-sample-chip").forEach(c => c.classList.remove("active"));
            fileInput.value = "";
        });

        function setCompareFile(file) {
            compareFile = file;
            const reader = new FileReader();
            reader.onload = e => {
                previewImg.src = e.target.result;
                previewWr.style.display = "block";
                dropZone.style.display = "none";
                runBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }

        // ── Run compare ───────────────────────────────────────────
        runBtn.addEventListener("click", runCompare);
    }

    async function runCompare() {
        if (!compareFile) return;

        const modelA = document.getElementById("compare-model-a-select")?.value;
        const modelB = document.getElementById("compare-model-b-select")?.value;
        const loading = document.getElementById("compare-loading");
        const grid    = document.getElementById("compare-results-grid");
        const runBtn  = document.getElementById("compare-run-btn");

        loading.style.display = "flex";
        grid.style.display = "none";
        runBtn.disabled = true;

        try {
            const fd = new FormData();
            fd.append("file", compareFile);
            if (modelA) fd.append("model_a", modelA);
            if (modelB) fd.append("model_b", modelB);
            fd.append("device", compareDevice);
            fd.append("compression", "4bit");
            fd.append("top_k", "5");

            const res  = await fetch(`${API_BASE}/compare`, { method: "POST", body: fd });
            const data = await res.json();

            if (!data.success) throw new Error(data.error || "API lỗi");

            renderCompareResults(data.results);
            grid.style.display = "grid";
        } catch(err) {
            alert(`Lỗi: ${err.message}`);
        } finally {
            loading.style.display = "none";
            runBtn.disabled = false;
        }
    }

    function renderCompareResults(results) {
        const ra = results.model_a;
        const rb = results.model_b;

        function fillPanel(res, prefix) {
            if (!res) return;
            const nameEl   = document.getElementById(`compare-result-${prefix}-name`);
            const timeEl   = document.getElementById(`compare-result-${prefix}-time`);
            const badgeEl  = document.getElementById(`compare-${prefix}-class-badge`);
            const cellName = document.getElementById(`compare-${prefix}-cell-name`);
            const confEl   = document.getElementById(`compare-${prefix}-confidence`);
            const barsEl   = document.getElementById(`compare-${prefix}-bars`);

            if (!res.success) {
                if (nameEl)  nameEl.textContent  = "Lỗi";
                if (badgeEl) badgeEl.textContent = "ERR";
                if (cellName) cellName.textContent = res.error || "Không thể tải model";
                return;
            }

            if (nameEl)  nameEl.textContent  = res.model_name || res.model_id || "—";
            if (timeEl)  timeEl.textContent  = `${res.inference_ms || 0}ms`;
            if (badgeEl) badgeEl.textContent = res.predicted_class || "?";
            if (cellName) cellName.textContent = res.class_name || "Unknown";
            if (confEl)  confEl.textContent  = `${((res.confidence || 0)*100).toFixed(1)}%`;

            // Confidence bars
            if (barsEl && res.top_predictions && res.top_predictions.length > 0) {
                barsEl.innerHTML = "";
                res.top_predictions.forEach((p, i) => {
                    const item = document.createElement("div");
                    item.className = "confidence-bar-item";
                    const pct = ((p.confidence || 0) * 100).toFixed(1);
                    const isTop = i === 0;
                    item.innerHTML = `
                        <span class="confidence-bar-label">${p.class || p.label || ""}</span>
                        <div class="confidence-bar-track">
                            <div class="confidence-bar-fill ${isTop ? 'top' : 'other'}"
                                 style="width: 0%" data-pct="${p.confidence || 0}"></div>
                        </div>
                        <span class="confidence-bar-pct">${pct}%</span>
                    `;
                    barsEl.appendChild(item);
                });
                // Animate bars
                setTimeout(() => {
                    barsEl.querySelectorAll(".confidence-bar-fill").forEach(bar => {
                        const pctVal = parseFloat(bar.dataset.pct || 0) * 100;
                        bar.style.width = `${Math.min(100, pctVal)}%`;
                    });
                }, 50);
            }
        }

        fillPanel(ra, "a");
        fillPanel(rb, "b");

        // ── Winner determination ──────────────────────────────────
        const winnerCard   = document.getElementById("compare-winner-card");
        const winnerLabel  = document.getElementById("compare-winner-label");
        const winnerDetail = document.getElementById("compare-winner-detail");

        if (ra && rb && ra.success && rb.success) {
            const classA = ra.predicted_class;
            const classB = rb.predicted_class;
            const confA  = ra.confidence || 0;
            const confB  = rb.confidence || 0;

            if (classA === classB) {
                // Cả 2 đồng ý
                winnerCard.className  = "compare-winner-card agree";
                winnerCard.querySelector(".fa-trophy").className = "fa-solid fa-handshake";
                winnerLabel.textContent  = "Đồng thuận";
                winnerDetail.textContent = `Cả 2 dự đoán ${classA}`;
            } else {
                // Khác nhau → chọn confidence cao hơn
                const winner = confA >= confB ? "A" : "B";
                winnerCard.className = "compare-winner-card";
                winnerCard.querySelector(".fa-handshake, .fa-trophy").className = "fa-solid fa-trophy";
                winnerLabel.textContent  = `Model ${winner} win`;
                winnerDetail.textContent = `${(Math.max(confA, confB)*100).toFixed(1)}% conf`;
            }
        } else {
            winnerLabel.textContent = "N/A";
            winnerDetail.textContent = "Lỗi model";
        }
    }

    // Startup
    initRouter();
    checkHealth();
    loadModels();
    setInterval(checkHealth, 10000);
});