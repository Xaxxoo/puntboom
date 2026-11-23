/* Predictions table fetch and render */

async function fetchPredictions(sport = null, valueOnly = false) {
    let url = '/api/predictions/';
    if (sport && sport !== 'all') {
        url = `/api/predictions/${sport}`;
    }
    const params = new URLSearchParams();
    if (valueOnly) params.append('value_only', 'true');
    if (params.toString()) url += '?' + params.toString();

    try {
        const response = await fetch(url);
        const data = await response.json();
        return data.predictions || [];
    } catch (err) {
        console.error('Failed to fetch predictions:', err);
        return [];
    }
}

async function fetchMatches(sport, status = 'all') {
    try {
        const response = await fetch(`/api/matches/${sport}?status=${status}`);
        const data = await response.json();
        return data.matches || [];
    } catch (err) {
        console.error('Failed to fetch matches:', err);
        return [];
    }
}
