/**
 * Fitsy - Node.js Express Server
 * Serves HTML, CSS, and JS with caching and compression
 */

const express = require('express');
const compression = require('compression');
const path = require('path');
const fs = require('fs');
const helmet = require('helmet');

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_API = process.env.BACKEND_API || 'http://localhost:8000';

// Security headers with CSP configured for inline event handlers
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'", "'unsafe-inline'"],
            scriptSrcAttr: ["'self'", "'unsafe-inline'"], // Allow inline event handlers (onclick, etc.)
            styleSrc: ["'self'", "'unsafe-inline'"],
            imgSrc: ["'self'", "data:", "https:", "http://localhost:8000"], // Allow images from backend API
            connectSrc: ["'self'", "http://localhost:8000", "ws://localhost:8000"],
            fontSrc: ["'self'", "data:"],
            objectSrc: ["'none'"],
            upgradeInsecureRequests: []
        }
    },
    crossOriginEmbedderPolicy: false
}));

// Enable gzip compression for all responses
app.use(compression({
    filter: (req, res) => {
        if (req.headers['x-no-compression']) {
            return false;
        }
        return compression.filter(req, res);
    },
    level: 6 // Balance between compression and speed
}));

// Middleware for setting cache headers
app.use((req, res, next) => {
    const filePath = req.path;

    // Static assets: cache for 30 days
    if (/\.(js|css|png|jpg|jpeg|gif|svg|woff|woff2|ttf|eot)$/i.test(filePath)) {
        // For development, disable caching to see changes immediately
        // res.set('Cache-Control', 'public, max-age=2592000, immutable');
        res.set('Cache-Control', 'no-cache, no-store, must-revalidate');
        res.set('ETag', generateETag(filePath));
    }
    // HTML files: cache for 1 hour
    else if (/\.html$/i.test(filePath)) {
        // res.set('Cache-Control', 'public, max-age=3600, must-revalidate');
        res.set('Cache-Control', 'no-cache, no-store, must-revalidate');
    }
    // API and everything else: no cache
    else {
        res.set('Cache-Control', 'no-cache, no-store, must-revalidate');
        res.set('Pragma', 'no-cache');
        res.set('Expires', '0');
    }

    // Set common security headers
    res.set('X-Content-Type-Options', 'nosniff');
    res.set('X-Frame-Options', 'DENY');
    res.set('X-XSS-Protection', '1; mode=block');

    next();
});

// Serve static files with caching
app.use(express.static(path.join(__dirname), {
    maxAge: '30d',
    etag: false // We're setting our own ETags
}));

// Proxy API requests to backend
app.use('/api', (req, res, next) => {
    const backendUrl = BACKEND_API + req.path.replace('/api', '');
    
    // Add CORS headers for API responses
    res.set('Access-Control-Allow-Origin', '*');
    res.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    
    // Forward the request
    const forwardReq = require('http').request(new URL(backendUrl + (req.url.split('?')[1] ? '?' + req.url.split('?')[1] : '')), {
        method: req.method,
        headers: req.headers
    }, (backendRes) => {
        res.writeHead(backendRes.statusCode, backendRes.headers);
        backendRes.pipe(res);
    });
    
    if (req.method !== 'GET') {
        req.pipe(forwardReq);
    } else {
        forwardReq.end();
    }
});

// Proxy static uploads from backend
app.use('/static/uploads', (req, res) => {
    const backendUrl = BACKEND_API + req.path;
    
    console.log(`Proxying image request: ${req.path} -> ${backendUrl}`);
    
    // Add CORS headers
    res.set('Access-Control-Allow-Origin', '*');
    res.set('Cache-Control', 'public, max-age=604800'); // Cache for 1 week
    
    const http = require('http');
    const urlObj = new URL(backendUrl);
    
    const options = {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname,
        method: 'GET',
        timeout: 10000
    };
    
    const proxyReq = http.request(options, (backendRes) => {
        console.log(`Backend response status: ${backendRes.statusCode}`);
        
        // Ensure we can load the image
        res.set('Content-Type', backendRes.headers['content-type'] || 'image/jpeg');
        res.set('Access-Control-Allow-Origin', '*');
        res.writeHead(backendRes.statusCode, backendRes.headers);
        backendRes.pipe(res);
    });
    
    proxyReq.on('error', (err) => {
        console.error('Error proxying image:', err);
        res.status(502).json({ error: 'Failed to load image from backend', details: err.message });
    });
    
    proxyReq.setTimeout(10000, () => {
        console.error('Proxy request timeout');
        proxyReq.destroy();
        res.status(504).json({ error: 'Proxy request timeout' });
    });
    
    proxyReq.end();
});

// Serve main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Fallback to index.html for client-side routing
app.get('*', (req, res) => {
    // Only serve HTML files
    if (req.path.endsWith('.html') || !req.path.includes('.')) {
        res.sendFile(path.join(__dirname, 'index.html'));
    } else {
        res.status(404).send('Not Found');
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({
        error: 'Internal Server Error',
        message: process.env.NODE_ENV === 'development' ? err.message : 'An error occurred'
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`
    ╔═════════════════════════════════════════╗
    ║  Fitsy - Node.js Frontend Server        ║
    ║  Running on: http://localhost:${PORT}       ║
    ║  Backend API: http://localhost:8000     ║
    ╚═════════════════════════════════════════╝
    `);
});

/**
 * Generate simple ETag for cache validation
 */
function generateETag(filePath) {
    const stat = fs.statSync(path.join(__dirname, filePath));
    return `"${stat.mtimeMs.toString(16)}"`;
}

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM signal received: closing HTTP server');
    app.close(() => {
        console.log('HTTP server closed');
        process.exit(0);
    });
});
