// API Configuration
const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000';
let wardrobeItems = {};
let recommendations = [];
let currentRecommendationIndex = 0;
let uploadedClothes = [];
let selectedWardrobeFiles = []; // Store accumulated files for upload
let currentTryOnMode = 'recommendations'; // 'recommendations' or 'upload'
let recommendationWebSocket = null;
let sessionId = null;

// Generate or retrieve session ID
function getSessionId() {
    if (!sessionId) {
        sessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        localStorage.setItem('fitsy_session_id', sessionId);
    }
    return sessionId;
}

// Page Navigation
function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });

    // Remove active class from all nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });

    // Show selected page
    const page = document.getElementById(pageId);
    if (page) {
        page.classList.add('active');
        
        // Add active class to nav link
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            if (link.textContent.toLowerCase().includes(pageId.split('-')[0])) {
                link.classList.add('active');
            }
        });

        // Load data if needed
        if (pageId === 'wardrobe') {
            loadWardrobe();
        }
    }
}

// Utility Functions
function showLoading(show = true) {
    document.getElementById('loadingSpinner').style.display = show ? 'flex' : 'none';
}

function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast show ${isError ? 'error' : ''}`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_URL}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showToast(`Error: ${error.message}`, true);
        return null;
    }
}

// Wardrobe Functions
async function loadWardrobe() {
    showLoading(true);
    const data = await fetchAPI('/items/all');
    
    if (data && data.items) {
        wardrobeItems = data.items;
        updateWardrobeStats(data);
        displayWardrobeItems();
        updateCategoryFilter();
    }
    
    showLoading(false);
}

function updateWardrobeStats(data) {
    document.getElementById('totalItems').textContent = data.total_items || 0;
    document.getElementById('totalCategories').textContent = Object.keys(data.categories || {}).length;
}

function displayWardrobeItems() {
    const grid = document.getElementById('wardrobeGrid');
    
    if (Object.keys(wardrobeItems).length === 0) {
        grid.innerHTML = '<div class="empty-state"><p>No items in your wardrobe yet</p><p>Upload clothing images to get started</p></div>';
        return;
    }

    grid.innerHTML = Object.entries(wardrobeItems).map(([name, item]) => `
        <div class="item-card">
            <button class="item-delete-btn" onclick="deleteItem('${name}', event)" title="Delete item">×</button>
            <div class="item-image">
                <img src="${item.url || ''}" alt="${name}" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22%3E%3Crect fill=%22%23e2e8f0%22 width=%22200%22 height=%22200%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 font-size=%2220%22 fill=%22%23999%22%3EImage%3C/text%3E%3C/svg%3E'">
            </div>
            <div class="item-info">
                <div class="item-name">${item.readable_name || name}</div>
                <div class="item-details">
                    <div>📁 ${item.category || 'Unknown'}</div>
                    <div>🎨 ${item.color || 'N/A'}</div>
                    <div>👕 ${item.style || 'N/A'}</div>
                </div>
            </div>
        </div>
    `).join('');
}

function updateCategoryFilter() {
    const categories = [...new Set(Object.values(wardrobeItems).map(item => item.category))];
    const select = document.getElementById('categoryFilter');
    
    select.innerHTML = '<option value="all">All Categories</option>' +
        categories.map(cat => `<option value="${cat}">${cat}</option>`).join('');
}

function filterWardrobe() {
    const category = document.getElementById('categoryFilter').value;
    const grid = document.getElementById('wardrobeGrid');
    
    let filtered = wardrobeItems;
    if (category !== 'all') {
        filtered = Object.entries(wardrobeItems)
            .filter(([, item]) => item.category === category)
            .reduce((acc, [key, val]) => {
                acc[key] = val;
                return acc;
            }, {});
    }

    grid.innerHTML = Object.entries(filtered).map(([name, item]) => `
        <div class="item-card">
            <button class="item-delete-btn" onclick="deleteItem('${name}', event)" title="Delete item">×</button>
            <div class="item-image">
                <img src="${item.url || ''}" alt="${name}">
            </div>
            <div class="item-info">
                <div class="item-name">${item.readable_name || name}</div>
                <div class="item-details">
                    <div>📁 ${item.category || 'Unknown'}</div>
                    <div>🎨 ${item.color || 'N/A'}</div>
                </div>
            </div>
        </div>
    `).join('');
}

// Handle wardrobe file preview
function handleWardrobePreview() {
    console.log('Handling wardrobe preview...');
    const input = document.getElementById('wardrobeUpload');
    const previewContainer = document.getElementById('wardrobeUploadPreview');
    
    if (!input || !previewContainer) {
        console.error('Preview elements not found');
        return;
    }
    
    // Don't clear existing previews - append instead
    // previewContainer.innerHTML = ''; 
    
    const newFiles = Array.from(input.files);
    console.log(`Selected ${newFiles.length} new files`);
    
    if (newFiles.length === 0) {
        return;
    }

    // Add new files to global array
    selectedWardrobeFiles = [...selectedWardrobeFiles, ...newFiles];
    console.log(`Total files selected: ${selectedWardrobeFiles.length}`);

    previewContainer.style.display = 'grid';

    newFiles.forEach(file => {
        if (!file.type.startsWith('image/')) return;

        const reader = new FileReader();
        reader.onload = function(e) {
            const imgContainer = document.createElement('div');
            imgContainer.style.position = 'relative';
            imgContainer.style.width = '100%';
            imgContainer.style.paddingTop = '100%'; // 1:1 Aspect Ratio
            imgContainer.style.overflow = 'hidden';
            imgContainer.style.borderRadius = '4px';
            imgContainer.style.border = '1px solid #e2e8f0';
            imgContainer.style.backgroundColor = '#f7fafc';

            const img = document.createElement('img');
            img.src = e.target.result;
            img.style.position = 'absolute';
            img.style.top = '0';
            img.style.left = '0';
            img.style.width = '100%';
            img.style.height = '100%';
            img.style.objectFit = 'cover';
            
            imgContainer.appendChild(img);
            previewContainer.appendChild(imgContainer);
        };
        reader.readAsDataURL(file);
    });

    // Reset input so the same file can be selected again if needed
    input.value = '';
}

function processWardrobeItems() {
    // Use global array instead of input.files
    if (selectedWardrobeFiles.length === 0) {
        showToast('Please select at least one image', true);
        return;
    }

    // Show wardrobe loading spinner
    document.getElementById('wardrobeLoadingSpinner').style.display = 'flex';
    showLoading(false); // Don't show global loading
    
    const formData = new FormData();
    for (let file of selectedWardrobeFiles) {
        formData.append('files', file);
    }

    fetch(`${API_URL}/items/add`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast(`✅ Added ${data.items_added} items to wardrobe!`);
            // Clear global array and previews
            selectedWardrobeFiles = [];
            const previewContainer = document.getElementById('wardrobeUploadPreview');
            if (previewContainer) {
                previewContainer.innerHTML = '';
                previewContainer.style.display = 'none';
            }
            loadWardrobe();
        } else {
            showToast('Failed to process items', true);
        }
    })
    .catch(error => {
        showToast(`Error: ${error.message}`, true);
    })
    .finally(() => {
        document.getElementById('wardrobeLoadingSpinner').style.display = 'none';
    });
}

// Delete individual item
async function deleteItem(itemName, event) {
    if (event) {
        event.stopPropagation(); // Prevent card click event
    }
    
    if (!confirm(`Are you sure you want to delete "${wardrobeItems[itemName]?.readable_name || itemName}"?`)) {
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_URL}/items/delete-by-names`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_names: [itemName]
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            showToast(`✅ Item deleted successfully`);
            loadWardrobe();
        } else {
            showToast('Failed to delete item', true);
        }
    } catch (error) {
        console.error('Delete error:', error);
        showToast(`Error: ${error.message}`, true);
    } finally {
        showLoading(false);
    }
}

// Delete all items
async function deleteAllItems() {
    const itemCount = Object.keys(wardrobeItems).length;
    
    if (itemCount === 0) {
        showToast('Your wardrobe is already empty', true);
        return;
    }
    
    if (!confirm(`Are you sure you want to delete ALL ${itemCount} items from your wardrobe? This action cannot be undone!`)) {
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_URL}/database/clear`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            showToast(`✅ All ${data.items_deleted} items deleted successfully`);
            loadWardrobe();
        } else {
            showToast('Failed to delete all items', true);
        }
    } catch (error) {
        console.error('Delete all error:', error);
        showToast(`Error: ${error.message}`, true);
    } finally {
        showLoading(false);
    }
}

// Recommendations Functions
async function getRecommendations() {
    const prompt = document.getElementById('prompt').value;
    
    if (!prompt) {
        showToast('Please describe your desired outfit', true);
        return;
    }

    // Show loading spinner for recommendations
    document.getElementById('recommendationsLoadingSpinner').style.display = 'flex';
    
    // Connect WebSocket if not already connected
    if (!recommendationWebSocket || recommendationWebSocket.readyState !== WebSocket.OPEN) {
        connectRecommendationWebSocket();
    }

    // Wait for WebSocket to be ready
    await new Promise(resolve => {
        if (recommendationWebSocket && recommendationWebSocket.readyState === WebSocket.OPEN) {
            resolve();
        } else {
            setTimeout(resolve, 1000);
        }
    });

    const payload = {
        prompt: prompt + " It should include a top, a bottom and footwear. You can also add accessory if it looks good.",
        num_recommendations: parseInt(document.getElementById('numRecommendations').value),
        user_preferences: {
            eye_color: document.getElementById('eyeColor').value,
            body_type: document.getElementById('bodyType').value,
            ethnicity: document.getElementById('ethnicity').value,
            temperature: document.getElementById('temperature').value || 'Not specified'
        }
    };

    // Send request through WebSocket
    if (recommendationWebSocket && recommendationWebSocket.readyState === WebSocket.OPEN) {
        recommendationWebSocket.send(JSON.stringify(payload));
    } else {
        showToast('WebSocket connection failed', true);
        document.getElementById('recommendationsLoadingSpinner').style.display = 'none';
    }
}

function connectRecommendationWebSocket() {
    const sessionId = getSessionId();
    const wsUrl = `${WS_URL}/ws/recommendations/${sessionId}`;
    
    recommendationWebSocket = new WebSocket(wsUrl);
    
    recommendationWebSocket.onopen = () => {
        console.log('WebSocket connected for recommendations');
        showToast('🔌 Connected to recommendation service');
    };
    
    recommendationWebSocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            if (data.error) {
                showToast(`Error: ${data.error}`, true);
            } else if (data.recommendations) {
                recommendations = data.recommendations;
                displayRecommendations();
                updateTryOnSelect();
                showToast('✨ New recommendations generated!');
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        } finally {
            // Hide loading spinner when done
            document.getElementById('recommendationsLoadingSpinner').style.display = 'none';
        }
    };
    
    recommendationWebSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        showToast('Connection error', true);
    };
    
    recommendationWebSocket.onclose = () => {
        console.log('WebSocket disconnected');
        showToast('🔴 Recommendation service disconnected - Chat history cleared');
    };
}

function displayRecommendations() {
    const results = document.getElementById('recommendationsResults');
    
    if (recommendations.length === 0) {
        results.innerHTML = '<div class="empty-state"><p>No recommendations yet</p></div>';
        return;
    }

    results.innerHTML = recommendations.map((rec, index) => `
        <div class="outfit-card">
            <div class="outfit-header">
                <h3>🎯 Outfit ${index + 1}</h3>
                <p class="outfit-recommendation">${rec.recommendation || 'Complete outfit'}</p>
            </div>
            <div class="outfit-body">
                <p class="outfit-reason"><strong>Why:</strong> ${rec.reason || 'Perfect match'}</p>
                
                <div class="outfit-items">
                    <h4>👕 Items:</h4>
                    <div class="items-list">
                        ${(rec.outfit_urls || []).map((url, i) => {
                            const readableName = rec.outfit_readable_names && rec.outfit_readable_names[i] 
                                ? rec.outfit_readable_names[i] 
                                : `Item ${i + 1}`;
                            return `
                                <div class="item-thumb">
                                    <img src="${url}" alt="${readableName}" title="${readableName}" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22150%22 height=%22150%22%3E%3Crect fill=%22%23e2e8f0%22 width=%22150%22 height=%22150%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 font-size=%2214%22 fill=%22%23999%22%3EImage Error%3C/text%3E%3C/svg%3E'">
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>

                ${rec.missing_items && rec.missing_items.length > 0 ? `
                    <p><strong>Missing:</strong> ${rec.missing_items.join(', ')}</p>
                ` : ''}

                <button class="btn btn-primary tryon-btn" onclick="selectOutfitForTryOn(${index})">
                    👤 Try This Outfit
                </button>
            </div>
        </div>
    `).join('');
}

function updateTryOnSelect() {
    const select = document.getElementById('recommendationSelect');
    select.innerHTML = recommendations.map((rec, index) => 
        `<option value="${index}">Outfit ${index + 1}: ${rec.recommendation || 'Outfit'}</option>`
    ).join('');
}

function selectOutfitForTryOn(index) {
    document.getElementById('recommendationSelect').value = index;
    showPage('tryon');
    showToast('Select your photo to try on this outfit');
}

// Try-On Functions
document.addEventListener('DOMContentLoaded', function() {
    // Set up wardrobe file upload change listener
    const wardrobeUpload = document.getElementById('wardrobeUpload');
    if (wardrobeUpload) {
        wardrobeUpload.addEventListener('change', handleWardrobePreview);
    }
    
    // Drag and drop for person image upload
    const uploadArea = document.getElementById('uploadArea');
    const personImageUpload = document.getElementById('personImageUpload');

    if (uploadArea && personImageUpload) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'rgba(102, 126, 234, 0.05)';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.backgroundColor = 'transparent';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'transparent';
            personImageUpload.files = e.dataTransfer.files;
            handlePersonImageUpload();
        });

        personImageUpload.addEventListener('change', handlePersonImageUpload);
    }
    
    // Initialize clothes upload handler
    initializeClothesUpload();
});

function handlePersonImageUpload() {
    const input = document.getElementById('personImageUpload');
    const file = input.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('personImagePreview');
            const previewImage = document.getElementById('previewImage');
            
            previewImage.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

// Try-On Tab Switching (kept for compatibility, now only supports recommendations)
function switchTryOnTab(tab) {
    // Only recommendations mode is supported
    currentTryOnMode = 'recommendations';
}

// Clothes Upload Handler
function initializeClothesUpload() {
    const clothesInput = document.getElementById('clothesUpload');
    const clothesUploadArea = document.getElementById('clothesUploadArea');
    
    if (!clothesInput || !clothesUploadArea) return;
    
    // Click to upload - directly trigger file input
    clothesUploadArea.addEventListener('click', (e) => {
        // Prevent propagation if clicking on upload area
        if (e.target === clothesUploadArea || e.target.closest('.upload-content')) {
            clothesInput.click();
        }
    });
    
    // Handle file selection
    clothesInput.addEventListener('change', (e) => {
        handleClothesUpload(e.target.files);
    });
    
    // Drag and drop
    clothesUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        clothesUploadArea.style.backgroundColor = 'rgba(102, 126, 234, 0.1)';
    });
    
    clothesUploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        clothesUploadArea.style.backgroundColor = '';
    });
    
    clothesUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        clothesUploadArea.style.backgroundColor = '';
        handleClothesUpload(e.dataTransfer.files);
    });
}

function handleClothesUpload(files) {
    // Don't clear the array - append to it instead
    let filesToAdd = [];
    let filesProcessed = 0;
    let totalImageFiles = 0;
    
    // Count total image files first
    for (let file of files) {
        if (file.type.startsWith('image/')) {
            totalImageFiles++;
        }
    }
    
    for (let file of files) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                filesToAdd.push({
                    name: file.name,
                    data: e.target.result,
                    file: file // Store the file object for FormData
                });
                filesProcessed++;
                
                // Add all files to the uploadedClothes array after all are loaded
                if (filesProcessed === totalImageFiles) {
                    uploadedClothes = uploadedClothes.concat(filesToAdd);
                    displayClothesPreview();
                    showToast(`✅ Added ${filesToAdd.length} clothing item(s)`);
                }
            };
            reader.readAsDataURL(file);
        }
    }
    
    if (totalImageFiles === 0) {
        showToast('Please select image files', true);
    }
}

function displayClothesPreview() {
    const preview = document.getElementById('uploadedClothesPreview');
    preview.innerHTML = '';
    
    if (uploadedClothes.length === 0) {
        preview.innerHTML = '<p style="text-align: center; color: #999; padding: 1rem;">No clothes uploaded yet</p>';
        return;
    }
    
    uploadedClothes.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'clothes-item';
        div.innerHTML = `
            <img src="${item.data}" alt="${item.name}">
            <span class="clothes-name">${item.name}</span>
            <button class="remove-btn" onclick="removeUploadedClothes(${index})">✕</button>
        `;
        preview.appendChild(div);
    });
}

function removeUploadedClothes(index) {
    uploadedClothes.splice(index, 1);
    displayClothesPreview();
}

async function generateTryOn() {
    const personImageUpload = document.getElementById('personImageUpload');

    // Validate person image
    if (!personImageUpload || !personImageUpload.files || !personImageUpload.files[0]) {
        showToast('Please upload your photo first', true);
        return;
    }

    // Validate recommendation selection
    const recommendationSelect = document.getElementById('recommendationSelect');
    const recommendationIndex = recommendationSelect.value;
    
    if (!recommendations || !recommendations[recommendationIndex]) {
        showToast('Please get recommendations first and select an outfit', true);
        return;
    }

    const imageNames = recommendations[recommendationIndex].image_names || [];
    
    if (!imageNames || imageNames.length === 0) {
        showToast('No outfit items found for this recommendation. Get new recommendations.', true);
        return;
    }

    document.getElementById('generateBtn').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'block';

    try {
        // Prepare form data with person image
        const formData = new FormData();
        formData.append('person_image', personImageUpload.files[0]);

        // Build URL with query parameters for image names
        const queryParams = imageNames.map(name => `recommendation_image_names=${encodeURIComponent(name)}`).join('&');
        const url = `${API_URL}/tryon/generate?${queryParams}`;

        console.log('Sending try-on request to:', url);
        console.log('Image names:', imageNames);

        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let errorDetail = `HTTP Error: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorDetail;
                console.error('Server error response:', errorData);
            } catch (e) {
                console.error('Could not parse error response');
            }
            throw new Error(errorDetail);
        }

        const data = await response.json();

        if (data && data.success) {
            // Display single try-on image
            const tryonImage = document.getElementById('tryonImage');
            if (tryonImage) {
                // Use the full URL from backend - the proxy will handle it
                const imageUrl = data.tryon_image_url;
                console.log('Loading image from URL:', imageUrl);
                
                // Try loading from local proxy first
                tryonImage.src = imageUrl;
                
                // Add error handler for image loading with fallback
                tryonImage.onerror = () => {
                    console.error('Failed to load image from direct URL, trying proxy...');
                    const imagePath = imageUrl.split('/').pop();
                    tryonImage.src = `/static/uploads/${imagePath}`;
                    
                    tryonImage.onerror = () => {
                        console.error('Failed to load try-on image from both sources');
                        showToast('Failed to load image. Please try again.', true);
                    };
                };
                
                // Also add onload for debugging
                tryonImage.onload = () => {
                    console.log('Image loaded successfully');
                };
            }
            
            // Display outfit items
            const outfitList = document.getElementById('outfitList');
            if (outfitList) {
                outfitList.innerHTML = (data.outfit_items || []).map(item => `<li>✓ ${item}</li>`).join('');
            }
            
            // Show results section
            const tryonResults = document.getElementById('tryonResults');
            if (tryonResults) {
                tryonResults.style.display = 'block';
                // Scroll to results
                tryonResults.scrollIntoView({ behavior: 'smooth' });
            }
            
            showToast('✨ Your AI try-on is ready!');
        } else {
            throw new Error(data.error || 'Failed to generate try-on image');
        }
    } catch (error) {
        console.error('Try-on generation error:', error);
        
        // Provide helpful error messages
        let userMessage = error.message;
        if (error.message.includes('Could not fetch any outfit images')) {
            userMessage = '❌ Could not find outfit images. Please upload clothes to your wardrobe first and get recommendations.';
        } else if (error.message.includes('Failed to fetch outfit images')) {
            userMessage = '❌ Error loading outfit images. Please ensure your wardrobe has items uploaded.';
        } else if (error.message.includes('500')) {
            userMessage = '❌ Server error. Please check the console logs and try again.';
        }
        
        showToast(userMessage, true);
    } finally {
        document.getElementById('generateBtn').style.display = 'block';
        document.getElementById('loadingSpinner').style.display = 'none';
    }
}

function downloadTryOn() {
    const img = document.getElementById('tryonImage');
    const link = document.createElement('a');
    link.href = img.src;
    link.download = 'fitsy-tryon.png';
    link.click();
    showToast('⬇️ Downloading try-on image...');
}

function shareTryOn() {
    const img = document.getElementById('tryonImage');
    if (navigator.share) {
        navigator.share({
            title: 'My FITSY Virtual Try-On',
            text: 'Check out my outfit with AI try-on!',
            url: window.location.href
        });
    } else {
        showToast('Share feature not available on your device', true);
    }
}

function generateAnotherTryon() {
    // Hide results but keep recommendations visible for testing
    document.getElementById('tryonResults').style.display = 'none';
    document.getElementById('generateBtn').style.display = 'block';
    
    // Select the first recommendation (or stay on current selection)
    const recommendationSelect = document.getElementById('recommendationSelect');
    if (recommendations && recommendations.length > 0) {
        recommendationSelect.value = 0;
    }
    
    // Scroll back to the recommendation selection
    recommendationSelect.scrollIntoView({ behavior: 'smooth' });
}

// Chat Functions
function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message) return;

    // Add user message to chat
    addChatMessage(message, 'user');
    input.value = '';

    // Simulate bot response
    setTimeout(() => {
        addChatMessage('Thanks for your message! I\'m analyzing your wardrobe...', 'bot');
    }, 500);
}

function handleChatKeyPress(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}

function addChatMessage(message, role) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    messageDiv.innerHTML = `<p>${escapeHtml(message)}</p>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function clearChat() {
    document.getElementById('chatMessages').innerHTML = '';
    addChatMessage('👋 Hi! I\'m your fashion assistant. Ask me anything about your wardrobe!', 'bot');
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    showPage('home');
});
