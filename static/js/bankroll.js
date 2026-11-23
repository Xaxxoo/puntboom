/* Bankroll calculator and tracker UI */

function bankrollApp() {
    return {
        bankroll: 1000,
        recommendations: [],
        summary: {
            total_staked: 0,
            total_return: 0,
            profit: 0,
            roi: 0,
        },
        pnlChart: null,

        async init() {
            await this.fetchSummary();
            await this.fetchPnL();
        },

        async fetchRecommendations() {
            try {
                const resp = await fetch(`/api/bankroll/recommendations?bankroll=${this.bankroll}`);
                const data = await resp.json();
                this.recommendations = data.recommendations || [];
            } catch (err) {
                console.error('Failed to fetch recommendations:', err);
            }
        },

        async fetchSummary() {
            try {
                const resp = await fetch('/api/bankroll/summary');
                const data = await resp.json();
                this.summary = {
                    total_staked: data.total_staked || 0,
                    total_return: data.total_return || 0,
                    profit: data.profit || 0,
                    roi: data.roi || 0,
                };
            } catch (err) {
                console.error('Failed to fetch summary:', err);
            }
        },

        async fetchPnL() {
            try {
                const resp = await fetch('/api/bankroll/pnl');
                const data = await resp.json();
                const history = data.history || [];

                if (history.length > 0) {
                    this.renderPnLChart(history);
                }
            } catch (err) {
                console.error('Failed to fetch P&L:', err);
            }
        },

        renderPnLChart(history) {
            const labels = history.map(h => h.date.split('T')[0]);
            const values = history.map(h => h.cumulative_pnl);

            const ctx = document.getElementById('pnl-chart');
            if (!ctx) return;

            if (this.pnlChart) {
                this.pnlChart.destroy();
            }

            this.pnlChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Cumulative P&L ($)',
                        data: values,
                        borderColor: values[values.length - 1] >= 0 ? '#10b981' : '#ef4444',
                        backgroundColor: values[values.length - 1] >= 0
                            ? 'rgba(16, 185, 129, 0.1)'
                            : 'rgba(239, 68, 68, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 2,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { intersect: false, mode: 'index' },
                    plugins: {
                        legend: { display: false },
                    },
                    scales: {
                        x: { display: true, grid: { color: '#1f2937' } },
                        y: {
                            display: true,
                            grid: { color: '#1f2937' },
                            title: { display: true, text: 'P&L ($)' },
                        },
                    },
                },
            });
        },
    };
}
