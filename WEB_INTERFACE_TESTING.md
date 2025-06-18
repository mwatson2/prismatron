# Web Interface Testing Guide

## 🚀 Current Status
Both backend and frontend servers are running successfully!

## 📍 Access Points

### Backend API Server
- **URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health
- **WebSocket**: ws://localhost:8000/ws

### Frontend Development Server  
- **URL**: http://localhost:3000
- **Network Access**: http://192.168.7.139:3000
- **mDNS Access**: http://prismatron.local:3000
- **Mobile Testing**: Use network URL or mDNS name on mobile devices

## 🧪 Testing Features

### 1. Home Page (http://localhost:3000/home)
- **LED Array Preview**: Simulated 60 LEDs with animation when playing
- **Playback Controls**: Play/Pause/Next/Previous buttons
- **System Status**: CPU, Memory, FPS, Uptime monitoring
- **Connection Status**: Real-time WebSocket connection indicator

### 2. Upload Page (http://localhost:3000/upload)
- **Drag & Drop**: Test with image/video files
- **File Validation**: Only allows JPG, PNG, GIF, MP4, MOV, WEBM
- **Progress Tracking**: Shows upload progress
- **Auto-Playlist**: Files automatically added to playlist

### 3. Effects Page (http://localhost:3000/effects)
- **Effect Categories**: Filter by color, animation, particle, etc.
- **Built-in Effects**: Rainbow Cycle, Color Wave, Sparkle
- **Custom Configuration**: Adjust parameters before adding
- **Add to Playlist**: Effects become playlist items

### 4. Playlist Page (http://localhost:3000/playlist)
- **Drag & Drop Reordering**: Reorder items by dragging
- **Playback Controls**: Shuffle, Repeat, Clear All
- **Item Management**: Play specific items, remove items
- **Statistics**: View total duration, item counts by type

### 5. Settings Page (http://localhost:3000/settings)
- **Brightness Control**: 0-100% slider with live preview
- **Frame Rate**: 15/24/30/60 FPS options
- **System Toggles**: Preview enabled, auto-start playlist
- **System Info**: Hardware details, version info

## 🔧 API Testing

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/api/health

# Get system status
curl http://localhost:8000/api/status

# Get playlist
curl http://localhost:8000/api/playlist

# Get effects
curl http://localhost:8000/api/effects

# Get settings
curl http://localhost:8000/api/settings

# Control playback
curl -X POST http://localhost:8000/api/control/play
curl -X POST http://localhost:8000/api/control/pause
```

## 📱 Mobile Testing

The interface is designed mobile-first. Test on your phone by:

1. **Connect to same network** as the server
2. **Visit**: http://192.168.7.139:3000 or http://prismatron.local:3000
3. **Install as PWA**: Use "Add to Home Screen" option
4. **Test touch interactions**: Navigation, drag & drop, controls

## 🌈 Visual Features

### Retro-Futurism Design
- **Neon Colors**: Cyan, Pink, Green, Orange, Purple accents
- **Glow Effects**: Text and border shadows with CSS animations
- **Grid Background**: Subtle LED grid pattern
- **Scan Line**: Animated scan line effect across top
- **Connection Status**: Animated LED indicator (top-right)

### Animations
- **Pulse**: For active elements and connection status
- **Glow**: For buttons and interactive elements
- **Flicker**: For offline/error states
- **Fade Transitions**: Smooth page transitions

## 🔧 Development Commands

```bash
# Backend (Terminal 1)
cd /mnt/dev/prismatron
python test_web_interface.py

# Frontend (Terminal 2)  
cd /mnt/dev/prismatron/src/web/frontend
npm run dev

# Production Build
npm run build

# Lint
npm run lint
```

## 🐛 Testing Scenarios

### Happy Path
1. Upload an image file
2. Add some effects to playlist
3. Use playback controls (play/pause/next)
4. Reorder playlist items
5. Adjust brightness setting
6. Verify real-time updates work

### Error Handling
1. Upload unsupported file type
2. Test with no internet connection
3. Try to play empty playlist
4. Test drag & drop with invalid files

### WebSocket Testing
1. Open multiple browser tabs
2. Make changes in one tab
3. Verify updates appear in other tabs
4. Test connection resilience

## 📊 Performance

### Current Build Stats
- **Bundle Size**: 367KB total (gzipped)
- **Load Time**: < 1 second on local network
- **PWA Ready**: Can be installed as mobile app
- **Real-time Updates**: WebSocket with auto-reconnection

## 🎯 Next Steps

1. **Test thoroughly** on mobile devices
2. **Verify WebSocket** real-time updates
3. **Upload test files** of various formats
4. **Check responsive design** at different screen sizes
5. **Test PWA installation** on mobile
6. **Validate accessibility** features

The web interface is fully functional and ready for comprehensive testing!