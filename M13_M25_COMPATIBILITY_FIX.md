# M13 Frontend → M25 Server Compatibility Fix

## Changes Made to `static/mind_server_frontendM13.html`

### 1. **Port Update: 7870 → 7871**
- **Previous**: `http://localhost:7870`
- **Current**: `http://localhost:7871`
- **Reason**: M25 server runs on port 7871 (E8_EC_PORT=7871)

### 2. **File Protocol Auto-Detection**
Added smart API_BASE_URL detection (lines 257-275):

```javascript
const API_BASE_URL = (() => {
    // Check for ?api=... query param override first
    const urlParam = new URLSearchParams(location.search).get('api');
    if (urlParam) return urlParam;
    
    // If opened as file://, connect to localhost:7871
    if (location.protocol === 'file:') {
        console.log('[M13] Detected file:// mode - connecting to http://localhost:7871');
        return 'http://localhost:7871';
    }
    
    // If served via HTTP, use same origin
    console.log('[M13] Detected HTTP mode - using same origin');
    return location.origin;
})();
```

**Benefits**:
- ✅ Works when double-clicking HTML file (`file://` protocol)
- ✅ Works when served by M25 HTTP server
- ✅ Supports `?api=http://custom:port` override
- ✅ Console logs show detected mode

## API Compatibility Verification

### M13 Frontend Requirements vs M25 Server APIs

| M13 Expects | M25 Provides | Status |
|------------|-------------|--------|
| `/api/telemetry/stream` | ✅ Line 33947 | ✅ Compatible |
| SSE `event: telemetry` | ✅ Line 33212, 33259 | ✅ Compatible |
| `telemetry.shells` | ✅ Line 30943 | ✅ Compatible |
| `telemetry.mood.intensity` | ✅ Line 30941 | ✅ Compatible |
| `telemetry.mood.coherence` | ✅ Line 30941 | ✅ Compatible |
| `telemetry.mood.entropy` | ✅ Line 30941 | ✅ Compatible |
| `telemetry.memory_count` | ✅ Line 30946 | ✅ Compatible |

### Telemetry Data Structure Match

**M13 JavaScript (lines 837-850)**:
```javascript
eventSource.addEventListener('telemetry', (event) => {
    const telemetry = JSON.parse(event.data);
    updateUI(telemetry);
    if (telemetry.shells) updateRotorState(telemetry.shells);
    
    if (telemetry.mood) {
        const intensity = telemetry.mood.intensity ?? 0.5;
        const coherence = telemetry.mood.coherence ?? 0.5;
        const entropy = telemetry.mood.entropy ?? 0.5;
    }
    
    const currentMemoryCount = telemetry.memory_count ?? state.lastMemoryCount;
});
```

**M25 Python (lines 30930-30950)**:
```python
telemetry = {
    "shells": shells_data,  # ✅
    "mood": self.mood.mood_vector,  # ✅ {intensity, coherence, entropy, ...}
    "memory_count": self.memory.graph_db.graph.number_of_nodes(),  # ✅
    ...
}
```

## Testing Instructions

### Method 1: Open as File
1. Navigate to `c:\Users\helio\Desktop\kaleidoscope\static\`
2. Double-click `mind_server_frontendM13.html`
3. Browser console should show: `[M13] Detected file:// mode - connecting to http://localhost:7871`
4. Ensure M25 server is running on port 7871
5. M13 HUD should connect and display telemetry

### Method 2: Served via M25
1. Start M25 server: `python e8_mind_server_M25.py` or `start_kaleidoscope_monolith.bat`
2. Navigate to: `http://localhost:7871/mind_server_frontendM13.html`
3. Browser console should show: `[M13] Detected HTTP mode - using same origin`
4. M13 HUD should connect and display telemetry

### Method 3: Custom API Override
Open with query param:
- `file:///.../mind_server_frontendM13.html?api=http://192.168.1.100:7871`
- `http://localhost:7871/mind_server_frontendM13.html?api=http://localhost:8888`

## Expected Behavior

- **Online Indicator**: "ONLINE" tag in green when connected
- **Rotating Shells**: 3D dimensional rotors spinning based on shell activity
- **Memory Etchings**: Flash animations when new memory nodes created
- **Mood Corona**: Bias effects based on mood intensity/coherence/entropy
- **Status Updates**: Real-time telemetry from M25 cognitive cycles

## No Breaking Changes

✅ All existing M13 features preserved
✅ Backward compatible with query param overrides
✅ Same telemetry event structure
✅ No changes to 3D visualization logic
✅ Only updated connection logic

## Files Modified

1. `static/mind_server_frontendM13.html` - Lines 257-275 (API_BASE_URL detection)

## Related Files

- `e8_mind_server_M25.py` - M25 server with telemetry SSE endpoint
- `static/index.html` - Main Observatory UI (already has file:// support)
- `start_kaleidoscope_monolith.bat` - Launches M25 on port 7871
