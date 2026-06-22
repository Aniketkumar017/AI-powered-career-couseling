// Main JavaScript for CareerAI

// Toast notifications
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    toast.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 250px;
        max-width: 400px;
        animation: fadeInDown 0.5s ease;
    `;
    toast.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

// Update progress bars
function updateProgressBars() {
    document.querySelectorAll('.match-bar').forEach(bar => {
        const fill = bar.querySelector('.match-fill');
        const percent = parseInt(fill.style.width) || 0;
        // Animate progress
        let current = 0;
        const interval = setInterval(() => {
            if (current >= percent) {
                clearInterval(interval);
                return;
            }
            current += 1;
            fill.style.width = current + '%';
        }, 10);
    });
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// Auto-save form data
function autoSaveForm() {
    const forms = document.querySelectorAll('form[data-auto-save]');
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, select, textarea');
        const formId = form.id || 'form';
        
        inputs.forEach(input => {
            input.addEventListener('change', debounce(() => {
                const data = new FormData(form);
                const obj = {};
                data.forEach((value, key) => obj[key] = value);
                localStorage.setItem(`autosave_${formId}`, JSON.stringify(obj));
            }, 500));
        });
        
        // Restore saved data
        const saved = localStorage.getItem(`autosave_${formId}`);
        if (saved) {
            try {
                const data = JSON.parse(saved);
                Object.keys(data).forEach(key => {
                    const input = form.querySelector(`[name="${key}"]`);
                    if (input) input.value = data[key];
                });
            } catch (e) {}
        }
    });
}

// Career search with suggestions
function setupCareerSearch() {
    const searchInput = document.getElementById('careerSearch');
    if (!searchInput) return;
    
    const suggestions = document.createElement('div');
    suggestions.className = 'suggestions-dropdown';
    suggestions.style.cssText = `
        position: absolute;
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        max-height: 200px;
        overflow-y: auto;
        z-index: 1000;
        width: 100%;
        display: none;
    `;
    searchInput.parentNode.style.position = 'relative';
    searchInput.parentNode.appendChild(suggestions);
    
    searchInput.addEventListener('input', debounce(async () => {
        const query = searchInput.value.trim();
        if (!query) {
            suggestions.style.display = 'none';
            return;
        }
        
        try {
            const response = await fetch(`/api/careers/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            if (data.success && data.data.length > 0) {
                suggestions.innerHTML = data.data.map(career => `
                    <div class="suggestion-item" style="padding: 8px 12px; cursor: pointer;">
                        ${career}
                    </div>
                `).join('');
                suggestions.style.display = 'block';
                
                suggestions.querySelectorAll('.suggestion-item').forEach(item => {
                    item.addEventListener('click', () => {
                        searchInput.value = item.textContent;
                        suggestions.style.display = 'none';
                        searchInput.form?.dispatchEvent(new Event('submit'));
                    });
                });
            } else {
                suggestions.style.display = 'none';
            }
        } catch (error) {
            console.error('Search error:', error);
            suggestions.style.display = 'none';
        }
    }, 300));
    
    document.addEventListener('click', (e) => {
        if (!searchInput.parentNode.contains(e.target)) {
            suggestions.style.display = 'none';
        }
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Update progress bars
    updateProgressBars();
    
    // Auto-save forms
    autoSaveForm();
    
    // Career search suggestions
    setupCareerSearch();
    
    // Add animation classes
    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    el.classList.add('animate-fade-in-up');
                }
            });
        });
        observer.observe(el);
    });
});