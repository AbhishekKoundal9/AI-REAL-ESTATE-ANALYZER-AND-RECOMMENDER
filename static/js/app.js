// ==========================================================================
// Main Application Controller for AI Real Estate Investment Analyzer
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadLocations();
    loadModelInfo();
    loadHistory();
    setupEventListeners();
});

let currentAnalysisResult = null;

// 1. Fetch Location Dropdowns
async function loadLocations() {
    try {
        const res = await fetch('/api/locations');
        const data = await res.json();
        const locations = data.locations || [];
        
        const locSelects = ['analyzeLocation', 'recLocation'];
        locSelects.forEach(selectId => {
            const selectEl = document.getElementById(selectId);
            if (selectEl) {
                selectEl.innerHTML = '';
                locations.forEach(loc => {
                    const opt = document.createElement('option');
                    opt.value = loc;
                    opt.textContent = loc;
                    if (loc === 'Whitefield') opt.selected = true;
                    selectEl.appendChild(opt);
                });
            }
        });
    } catch (err) {
        console.error('Failed to load locations:', err);
    }
}

// 2. Fetch ML Model Performance Info
async function loadModelInfo() {
    try {
        const res = await fetch('/api/model-info');
        const data = await res.json();
        
        const lrR2 = document.getElementById('lrR2');
        const rfR2 = document.getElementById('rfR2');
        const bestModelBadge = document.getElementById('bestModelBadge');
        
        if (lrR2 && data.linear_regression) {
            lrR2.textContent = `${(data.linear_regression.r2_score * 100).toFixed(1)}%`;
        }
        if (rfR2 && data.random_forest) {
            rfR2.textContent = `${(data.random_forest.r2_score * 100).toFixed(1)}%`;
        }
        if (bestModelBadge) {
            bestModelBadge.textContent = data.best_model || 'Random Forest Regressor';
        }
        
        renderModelComparisonChart(data);
    } catch (err) {
        console.error('Failed to load model info:', err);
    }
}

// 3. Event Listeners Setup
function setupEventListeners() {
    // Analyze Form Submission
    const analyzeForm = document.getElementById('analyzeForm');
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', (e) => {
            e.preventDefault();
            runAgenticAnalysis();
        });
    }

    // Recommendation Form Submission
    const recForm = document.getElementById('recommendationForm');
    if (recForm) {
        recForm.addEventListener('submit', (e) => {
            e.preventDefault();
            fetchRecommendations();
        });
    }
}

// 4. One-Click Basic Agentic AI Pipeline
async function runAgenticAnalysis() {
    const location = document.getElementById('analyzeLocation').value;
    const total_sqft = parseFloat(document.getElementById('analyzeSqft').value);
    const bhk = parseInt(document.getElementById('analyzeBhk').value);
    const bath = parseInt(document.getElementById('analyzeBath').value);
    const balcony = parseInt(document.getElementById('analyzeBalcony').value);
    const budget = parseFloat(document.getElementById('analyzeBudget').value) || null;

    if (!location || !total_sqft || !bhk) {
        alert('Please fill in all required property specifications.');
        return;
    }

    // Show Loading Modal with Step Progress Animation
    const modalEl = document.getElementById('agentModal');
    const modal = new bootstrap.Modal(modalEl);
    modal.show();

    const steps = [
        'step-1', 'step-2', 'step-3', 'step-4', 
        'step-5', 'step-6', 'step-7', 'step-8'
    ];

    // Reset steps UI
    steps.forEach(s => {
        const el = document.getElementById(s);
        if (el) {
            el.className = 'agent-step-item';
            const icon = el.querySelector('i');
            if (icon) icon.className = 'fas fa-circle-notch fa-spin text-purple me-2';
        }
    });

    // Simulate Agent execution step feedback sequentially
    for (let i = 0; i < steps.length; i++) {
        const el = document.getElementById(steps[i]);
        if (el) {
            el.classList.add('active');
        }
        await new Promise(r => setTimeout(r, 220));
        if (el) {
            el.classList.remove('active');
            el.classList.add('completed');
            const icon = el.querySelector('i');
            if (icon) icon.className = 'fas fa-check-circle text-success me-2';
        }
    }

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                location,
                total_sqft,
                bhk,
                bath,
                balcony,
                budget
            })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        currentAnalysisResult = data;

        // Hide Modal
        setTimeout(() => {
            modal.hide();
            displayAnalysisResults(data);
            loadHistory();
        }, 400);

    } catch (err) {
        modal.hide();
        alert(`Analysis Error: ${err.message}`);
    }
}

// 5. Display Analysis Results
function displayAnalysisResults(data) {
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Set Values
    document.getElementById('resPredictedPrice').textContent = `₹${data.predicted_price_lakhs.toFixed(2)} Lakhs`;
    document.getElementById('resPriceInr').textContent = `(Approx. ${data.predicted_price_inr})`;
    document.getElementById('resPricePerSqft').textContent = `₹${data.price_per_sqft.toLocaleString()}/sq.ft`;
    
    document.getElementById('resRoi').textContent = `${data.roi_percent}%`;
    document.getElementById('resRentalYield').textContent = `${data.rental_yield_percent}%`;
    document.getElementById('resScore').textContent = `${data.investment_score}/100`;

    // Risk Badge
    const riskBadge = document.getElementById('resRiskBadge');
    riskBadge.textContent = data.risk_level;
    riskBadge.className = 'badge ';
    if (data.risk_level === 'Low Risk') riskBadge.classList.add('badge-risk-low');
    else if (data.risk_level === 'Medium Risk') riskBadge.classList.add('badge-risk-medium');
    else riskBadge.classList.add('badge-risk-high');

    // Recommendation Badge & Rationale
    const recBadge = document.getElementById('resRecBadge');
    recBadge.textContent = data.recommendation;
    recBadge.className = 'badge ';
    if (data.recommendation === 'BUY') recBadge.classList.add('badge-buy');
    else if (data.recommendation === 'HOLD') recBadge.classList.add('badge-hold');
    else recBadge.classList.add('badge-avoid');

    // Recommendation Reasons List
    const reasonsList = document.getElementById('resReasonsList');
    reasonsList.innerHTML = '';
    (data.reasons || []).forEach(r => {
        const li = document.createElement('li');
        li.className = 'mb-1 text-light';
        li.innerHTML = `<i class="fas fa-check text-purple me-2"></i>${r}`;
        reasonsList.appendChild(li);
    });

    // AI Insights List
    const insightsList = document.getElementById('resInsightsList');
    insightsList.innerHTML = '';
    (data.insights || []).forEach(ins => {
        const p = document.createElement('p');
        p.className = 'mb-2 text-secondary';
        p.innerHTML = `<i class="fas fa-lightbulb text-cyan me-2"></i>${ins}`;
        insightsList.appendChild(p);
    });

    // Future Valuations
    document.getElementById('resFuture1Yr').textContent = `₹${data.future_prices.price_1yr.toFixed(2)} Lakhs`;
    document.getElementById('resFuture3Yr').textContent = `₹${data.future_prices.price_3yr.toFixed(2)} Lakhs`;

    // PDF Download Button setup
    const pdfBtn = document.getElementById('downloadPdfBtn');
    if (pdfBtn) {
        pdfBtn.onclick = () => {
            window.open(`/api/pdf/${data.pdf_filename}`, '_blank');
        };
    }

    // Render Dashboard Charts
    updatePriceComparisonChart(data.predicted_price_lakhs, (data.avg_price_per_sqft * data.input_data.total_sqft) / 100000.0);
    updateGrowthProjectionChart(data.predicted_price_lakhs, data.future_prices.price_1yr, data.future_prices.price_3yr);
    updateInvestmentMetricsChart(data.roi_percent, data.rental_yield_percent, data.investment_score);
}

// 6. Property Recommendation Query
async function fetchRecommendations() {
    const budget = parseFloat(document.getElementById('recBudget').value);
    const location = document.getElementById('recLocation').value;
    const bhk = parseInt(document.getElementById('recBhk').value);

    if (!budget) {
        alert('Please enter your target investment budget.');
        return;
    }

    try {
        const res = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ budget, location, bhk })
        });

        const data = await res.json();
        renderRecommendations(data.recommendations || []);
    } catch (err) {
        alert(`Recommendation fetch failed: ${err.message}`);
    }
}

function renderRecommendations(items) {
    const container = document.getElementById('recommendationsGrid');
    container.innerHTML = '';

    if (items.length === 0) {
        container.innerHTML = `
            <div class="col-12 text-center text-muted py-4">
                <i class="fas fa-search fa-2x me-2"></i> No matching dataset properties found for the specified criteria.
            </div>`;
        return;
    }

    items.forEach(item => {
        const cardHtml = `
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="glass-card h-100 p-4 position-relative">
                    <div class="d-flex justify-content-between align-items-start mb-3">
                        <span class="badge bg-purple-glow">${item.match_score}% Match</span>
                        <span class="text-cyan fw-bold fs-5">₹${item.price_lakhs.toFixed(2)} Lakhs</span>
                    </div>
                    <h5 class="fw-bold mb-2">${item.location}</h5>
                    <div class="text-secondary small mb-3">
                        <span class="me-3"><i class="fas fa-vector-square me-1 text-purple"></i> ${item.total_sqft} sq.ft</span>
                        <span class="me-3"><i class="fas fa-bed me-1 text-cyan"></i> ${item.bhk} BHK</span>
                        <span><i class="fas fa-bath me-1 text-muted"></i> ${item.bath} Bath</span>
                    </div>
                    <div class="border-top border-glass pt-3 d-flex justify-content-between align-items-center">
                        <small class="text-muted">Kaggle Verified Property</small>
                        <button class="btn btn-sm btn-outline-light" onclick="quickAnalyze('${item.location}', ${item.total_sqft}, ${item.bhk}, ${item.bath}, ${item.balcony})">
                            <i class="fas fa-microchip me-1 text-cyan"></i> Analyze
                        </button>
                    </div>
                </div>
            </div>`;
        container.innerHTML += cardHtml;
    });
}

function quickAnalyze(location, sqft, bhk, bath, balcony) {
    // Switch to analyze tab & fill inputs
    const tabEl = document.querySelector('button[data-bs-target="#analyze-tab"]');
    if (tabEl) {
        const tab = new bootstrap.Tab(tabEl);
        tab.show();
    }
    document.getElementById('analyzeLocation').value = location;
    document.getElementById('analyzeSqft').value = sqft;
    document.getElementById('analyzeBhk').value = bhk;
    document.getElementById('analyzeBath').value = bath;
    document.getElementById('analyzeBalcony').value = balcony;

    runAgenticAnalysis();
}

// 7. Load History from SQLite
async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        const records = data.history || [];

        const tbody = document.getElementById('historyTableBody');
        if (!tbody) return;
        tbody.innerHTML = '';

        if (records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" class="text-center text-muted py-4">No analysis history recorded yet. Run your first property analysis above!</td></tr>`;
            return;
        }

        records.forEach(rec => {
            const dateStr = new Date(rec.created_at || Date.now()).toLocaleString();
            const tr = document.createElement('tr');

            const recBadgeClass = rec.recommendation === 'BUY' ? 'badge-buy' : (rec.recommendation === 'HOLD' ? 'badge-hold' : 'badge-avoid');

            tr.innerHTML = `
                <td>#${rec.id}</td>
                <td><small class="text-muted">${dateStr}</small></td>
                <td class="fw-bold">${rec.location}</td>
                <td>${rec.sqft} sqft | ${rec.bhk} BHK</td>
                <td class="text-cyan fw-bold">₹${rec.predicted_price.toFixed(2)} L</td>
                <td>${rec.roi}%</td>
                <td><span class="badge ${recBadgeClass}">${rec.recommendation}</span></td>
                <td>
                    ${rec.pdf_filename ? `<a href="/api/pdf/${rec.pdf_filename}" target="_blank" class="btn btn-sm btn-outline-cyan"><i class="fas fa-file-pdf me-1"></i> PDF</a>` : '-'}
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteHistory(${rec.id})"><i class="fas fa-trash"></i></button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

async function deleteHistory(id) {
    if (!confirm(`Delete history record #${id}?`)) return;
    try {
        await fetch(`/api/history/${id}`, { method: 'DELETE' });
        loadHistory();
    } catch (err) {
        alert(`Delete failed: ${err.message}`);
    }
}
