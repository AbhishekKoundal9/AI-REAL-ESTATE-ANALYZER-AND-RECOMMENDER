// ==========================================================================
// Chart.js Manager for AI Real Estate Investment Analyzer
// ==========================================================================

let priceComparisonChartInstance = null;
let growthProjectionChartInstance = null;
let investmentMetricsChartInstance = null;
let modelComparisonChartInstance = null;

function initCharts() {
    Chart.defaults.color = '#94A3B8';
    Chart.defaults.font.family = "'Inter', sans-serif";
}

function updatePriceComparisonChart(predictedPrice, avgPrice) {
    const ctx = document.getElementById('priceComparisonChart');
    if (!ctx) return;

    if (priceComparisonChartInstance) {
        priceComparisonChartInstance.destroy();
    }

    priceComparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Predicted Property Price', 'Area Market Average'],
            datasets: [{
                label: 'Price (in Lakhs ₹)',
                data: [predictedPrice, avgPrice],
                backgroundColor: [
                    'rgba(139, 92, 246, 0.75)',
                    'rgba(6, 182, 212, 0.75)'
                ],
                borderColor: [
                    '#8B5CF6',
                    '#06B6D4'
                ],
                borderWidth: 1.5,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` Price: ₹${context.raw.toFixed(2)} Lakhs`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

function updateGrowthProjectionChart(currentPrice, price1Yr, price3Yr) {
    const ctx = document.getElementById('growthProjectionChart');
    if (!ctx) return;

    if (growthProjectionChartInstance) {
        growthProjectionChartInstance.destroy();
    }

    growthProjectionChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Current Value', '1-Year Projection', '3-Year Projection'],
            datasets: [{
                label: 'Property Value (Lakhs ₹)',
                data: [currentPrice, price1Yr, price3Yr],
                fill: true,
                backgroundColor: 'rgba(139, 92, 246, 0.15)',
                borderColor: '#A855F7',
                borderWidth: 3,
                pointBackgroundColor: '#06B6D4',
                pointBorderColor: '#FFFFFF',
                pointRadius: 6,
                pointHoverRadius: 8,
                tension: 0.35
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` Estimated Price: ₹${context.raw.toFixed(2)} Lakhs`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

function updateInvestmentMetricsChart(roi, rentalYield, score) {
    const ctx = document.getElementById('investmentMetricsChart');
    if (!ctx) return;

    if (investmentMetricsChartInstance) {
        investmentMetricsChartInstance.destroy();
    }

    investmentMetricsChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['ROI Potential (%)', 'Rental Yield (%)', 'Investment Score (/100)'],
            datasets: [{
                data: [roi, rentalYield * 10, score],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(6, 182, 212, 0.8)',
                    'rgba(139, 92, 246, 0.8)'
                ],
                borderColor: '#0F172A',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94A3B8', padding: 15 }
                }
            },
            cutout: '65%'
        }
    });
}

function renderModelComparisonChart(metrics) {
    const ctx = document.getElementById('modelComparisonChart');
    if (!ctx || !metrics) return;

    if (modelComparisonChartInstance) {
        modelComparisonChartInstance.destroy();
    }

    const lrR2 = (metrics.linear_regression?.r2_score || 0.68) * 100;
    const rfR2 = (metrics.random_forest?.r2_score || 0.85) * 100;
    
    const lrRmse = metrics.linear_regression?.rmse || 22.0;
    const rfRmse = metrics.random_forest?.rmse || 14.0;

    modelComparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['R² Accuracy Score (%)', 'RMSE Error (Lower is better)'],
            datasets: [
                {
                    label: 'Linear Regression',
                    data: [lrR2, lrRmse],
                    backgroundColor: 'rgba(6, 182, 212, 0.7)',
                    borderColor: '#06B6D4',
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: 'Random Forest Regressor',
                    data: [rfR2, rfRmse],
                    backgroundColor: 'rgba(139, 92, 246, 0.85)',
                    borderColor: '#8B5CF6',
                    borderWidth: 1,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#94A3B8' }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}
