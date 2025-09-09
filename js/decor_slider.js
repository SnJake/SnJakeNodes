import { app } from "../../../scripts/app.js";
// Toggle to enable/disable planar waves visual
const SJ_WAVES_ENABLED = false;
// Toggle to enable/disable edge pulses (from sides to center)
const SJ_EDGE_PULSE_ENABLED = false;

// Universal decorative silver slider for all SnJake nodes (titles prefixed with "😎")
app.registerExtension({
    name: "SnJake.DecorSlider",
    async nodeCreated(node) {
        // Only target SnJake nodes that use the 😎 prefix in their title
        const title = (node && (node.title || "")).toString();
        const isSnJake = title.startsWith("😎");
        if (!isSnJake) return;

        // Avoid double-initialization if a node-specific script already added it
        if (node.__sjake_anim_init) return;

        node.__sjake_anim_init = true;

        // Keep original hooks
        const prevOnDrawForeground = node.onDrawForeground?.bind(node);
        const prevOnRemoved = node.onRemoved?.bind(node);

        node.__sjake_phase_offset = Math.random() * Math.PI * 2;
        node.__sjake_waves = [];
        node.__sjake_last_time = performance.now() / 1000;
        node.__sjake_next_pulse = node.__sjake_last_time + 2 + Math.random() * 4; // rare pulses

        node.onDrawForeground = function (ctx) {
            // Call original foreground drawing first
            if (prevOnDrawForeground) prevOnDrawForeground(ctx);

            const w = this.size?.[0] ?? 0;
            const h = this.size?.[1] ?? 0;
            if (!w || !h) return;

            // Skip when collapsed/minimized
            if (this.flags && (this.flags.collapsed || this.flags.minimized)) return;

            const pad = 8;
            const trackH = 6;
            const radius = 3;
            const x = pad;
            // Draw slightly below the node so it never overlaps widgets
            const y = h + 6; // gap below the node
            const trackW = Math.max(0, w - pad * 2);

            // Time-based position (ping-pong 0..1)
            const now = performance.now() / 1000;
            const t = now * 1.2 + (this.__sjake_phase_offset || 0);
            const pingpong = 0.5 * (1 + Math.sin(t));
            const thumbW = Math.max(16, Math.min(28, trackW * 0.2));
            const thumbX = x + pingpong * (trackW - thumbW);

            // Theme-aware colors derived from node scheme (auto_colors)
            const theme = computeDecorColors(this);

            // Draw track with edge-to-center gradient and a subtle outline
            ctx.save();
            roundRect(ctx, x, y, trackW, trackH, radius);
            const trackGrad = ctx.createLinearGradient(x, y, x + trackW, y);
            trackGrad.addColorStop(0.0, theme.trackEdge);
            trackGrad.addColorStop(0.5, theme.trackCenter);
            trackGrad.addColorStop(1.0, theme.trackEdge);
            ctx.fillStyle = trackGrad;
            ctx.fill();
            // subtle inner stroke to make it a bit more noticeable
            ctx.strokeStyle = theme.trackStroke;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();

            // Optional planar waves (disabled by default)
            if (SJ_WAVES_ENABLED) {
                const dt = Math.min(0.1, Math.max(0, now - (this.__sjake_last_time || now)));
                this.__sjake_last_time = now;
                if (now >= (this.__sjake_next_pulse || now)) {
                    spawnWaves(this, thumbX + thumbW * 0.5, x, x + trackW, theme);
                    this.__sjake_next_pulse = now + 3 + Math.random() * 4; // next pulse in 3..7s
                }
                drawWaves(ctx, this, x, y, trackW, trackH, theme, dt);
            }

            // Edge-to-center pulse from both sides (rare and gentle)
            if (SJ_EDGE_PULSE_ENABLED) {
                handleEdgePulses(this, now, x, y, trackW, trackH, ctx, theme);
            }

            // Slight pulsing of the thumb (subtle)
            const pulse = 1 + 0.06 * Math.sin(now * 0.9 + (this.__sjake_phase_offset || 0));
            const thumbH = (trackH + 1) * pulse;
            const thumbY = y - 0.5 - (thumbH - (trackH + 1)) * 0.5;

            // Thumb with edge-to-center gradient based on theme
            const grad = ctx.createLinearGradient(thumbX, y, thumbX + thumbW, y);
            grad.addColorStop(0.0, theme.thumbEdge);
            grad.addColorStop(0.5, theme.thumbCenter);
            grad.addColorStop(1.0, theme.thumbEdge);

            // Shadow + fill
            ctx.save();
            ctx.shadowColor = "rgba(0,0,0,0.25)";
            ctx.shadowBlur = 2;
            ctx.shadowOffsetY = 1;
            roundRect(ctx, thumbX, thumbY, thumbW, thumbH, radius);
            ctx.fillStyle = grad;
            ctx.fill();
            ctx.restore();

            // Top highlight line for a subtle sheen
            ctx.save();
            ctx.globalAlpha = 0.35;
            ctx.beginPath();
            ctx.moveTo(thumbX + 1, thumbY + 1);
            ctx.lineTo(thumbX + thumbW - 1, thumbY + 1);
            ctx.strokeStyle = theme.thumbHighlight;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();
        };

        // Lightweight animation loop to keep canvas fresh while node exists
        const ensureRedraw = () => {
            const canvas = app?.canvas;
            if (!canvas) return;
            if (typeof canvas.setDirty === "function") canvas.setDirty(true, true);
            else if (typeof canvas.draw === "function") canvas.draw(true, true);
        };

        const animate = () => {
            if (!node.graph) return; // stop when node is detached
            ensureRedraw();
            node.__sjake_anim_frame = requestAnimationFrame(animate);
        };
        node.__sjake_anim_frame = requestAnimationFrame(animate);

        // Cleanup when removed
        node.onRemoved = function () {
            if (prevOnRemoved) prevOnRemoved();
            if (this.__sjake_anim_frame) cancelAnimationFrame(this.__sjake_anim_frame);
            this.__sjake_anim_frame = null;
        };
    }
});

// Helper: rounded rectangle path
function roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w * 0.5, h * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.lineTo(x + w - rr, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
    ctx.lineTo(x + w, y + h - rr);
    ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    ctx.lineTo(x + rr, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
    ctx.lineTo(x, y + rr);
    ctx.quadraticCurveTo(x, y, x + rr, y);
    ctx.closePath();
}

// ---- Theming helpers (derived from auto_colors scheme) ----
function computeDecorColors(node) {
    // Prefer colors already applied by AutoColors if present
    const bg = safeColor(node?.bgcolor, "#41414a");
    const base = safeColor(node?.color, "#2e2e36");
    const title = safeColor(node?.constructor?.title_text_color, "#e5e9f0");

    // Track gradient (edge -> center -> edge) derived from bg/base with slight alpha
    const trackEdge = setAlphaHex(lightenHex(mixHex(base, title, 0.25), 0.08), 0.75);
    const trackCenter = setAlphaHex(mixHex(bg, base, 0.35), 0.55);
    const trackStroke = setAlphaHex(darkenHex(base, 0.35), 0.45);

    // Thumb gradient (edge -> center -> edge): edges a bit lighter, center slightly darker
    const thumbEdge = lightenHex(base, 0.32);
    const thumbCenter = darkenHex(base, 0.10);

    // Wave color: use title text tint with transparency
    const wave = setAlphaHex(title, 0.22);
    const waveStrong = setAlphaHex(title, 0.35);

    return {
        trackEdge,
        trackCenter,
        trackStroke,
        thumbEdge,
        thumbCenter,
        thumbHighlight: setAlphaHex("#ffffff", 0.35),
        wave,
        waveStrong
    };
}

function safeColor(v, fallback) {
    if (typeof v === "string" && /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(v)) return v;
    return fallback;
}

function hexToRgb(hex) {
    hex = hex.replace(/^#/, "");
    if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
    const int = parseInt(hex, 16);
    return { r: (int >> 16) & 255, g: (int >> 8) & 255, b: int & 255 };
}
function rgbToHex(r, g, b) {
    const to = (n) => Math.max(0, Math.min(255, n | 0)).toString(16).padStart(2, "0");
    return `#${to(r)}${to(g)}${to(b)}`;
}
function mixHex(a, b, t, alpha = 1) {
    const A = hexToRgb(a), B = hexToRgb(b);
    const r = A.r + (B.r - A.r) * t;
    const g = A.g + (B.g - A.g) * t;
    const bl = A.b + (B.b - A.b) * t;
    const base = rgbToHex(r, g, bl);
    return setAlphaHex(base, alpha);
}
function lightenHex(hex, amt) {
    const c = hexToRgb(hex);
    const k = amt || 0;
    return rgbToHex(
        c.r + (255 - c.r) * k,
        c.g + (255 - c.g) * k,
        c.b + (255 - c.b) * k
    );
}
function darkenHex(hex, amt) {
    const c = hexToRgb(hex);
    const k = amt || 0;
    return rgbToHex(
        c.r * (1 - k),
        c.g * (1 - k),
        c.b * (1 - k)
    );
}
function setAlphaHex(hex, alpha) {
    const { r, g, b } = hexToRgb(hex);
    const a = Math.max(0, Math.min(1, alpha));
    // Return rgba() for canvas fillStyle support with alpha
    return `rgba(${r},${g},${b},${a})`;
}

// ---- Wave system ----
function spawnWaves(node, centerX, minX, maxX, theme) {
    // Two planar wavefronts moving left and right from the thumb
    const speed = 80 + Math.random() * 60; // px/s
    const width = 8; // visual width of wavefront
    const life = 2.2; // seconds
    const now = performance.now() / 1000;
    const base = { start: now, x: centerX, speed, width, life, color: theme.wave, strong: theme.waveStrong };
    node.__sjake_waves.push({ ...base, dir: -1, minX, maxX });
    node.__sjake_waves.push({ ...base, dir: +1, minX, maxX });
    // Cap number of waves to avoid buildup
    if (node.__sjake_waves.length > 12) node.__sjake_waves.splice(0, node.__sjake_waves.length - 12);
}

function drawWaves(ctx, node, x, y, w, h, theme, dt) {
    const now = performance.now() / 1000;
    const waves = node.__sjake_waves || [];
    for (let i = waves.length - 1; i >= 0; i--) {
        const wave = waves[i];
        const t = now - wave.start;
        if (t > wave.life) { waves.splice(i, 1); continue; }
        const eased = t / wave.life;
        const alphaScale = 1 - eased;
        const curX = wave.x + wave.dir * wave.speed * t;
        if (curX < wave.minX - 12 || curX > wave.maxX + 12) { waves.splice(i, 1); continue; }

        // Draw a soft vertical stripe (planar wavefront)
        const stripeW = wave.width * (1 + 0.2 * eased);
        const gx0 = curX - stripeW * 0.5;
        const gx1 = curX + stripeW * 0.5;
        const grad = ctx.createLinearGradient(gx0, 0, gx1, 0);
        grad.addColorStop(0.0, "rgba(0,0,0,0)");
        grad.addColorStop(0.5, blendAlpha(wave.strong, alphaScale));
        grad.addColorStop(1.0, "rgba(0,0,0,0)");

        ctx.save();
        ctx.beginPath();
        roundRect(ctx, x, y, w, h, 3);
        ctx.clip();
        ctx.fillStyle = grad;
        ctx.fillRect(gx0, y - 2, stripeW, h + 4);
        ctx.restore();
    }
}

function blendAlpha(rgba, scale) {
    // rgba(r,g,b,a) -> same r,g,b with a * scale
    const m = rgba.match(/^rgba\((\d+),(\d+),(\d+),(\d*\.?\d+)\)$/);
    if (!m) return rgba;
    const r = +m[1], g = +m[2], b = +m[3], a = +m[4];
    const na = Math.max(0, Math.min(1, a * scale));
    return `rgba(${r},${g},${b},${na})`;
}

// ---- Edge pulse system (from sides to center) ----
function handleEdgePulses(node, now, x, y, w, h, ctx, theme) {
    // Lazy-init
    if (!node.__sjake_edge_pulses) node.__sjake_edge_pulses = [];
    if (node.__sjake_next_sidepulse == null) {
        node.__sjake_next_sidepulse = now + 5 + Math.random() * 7; // first pulse in 5..12s
    }

    // Spawn new pulse occasionally (not often)
    if (now >= node.__sjake_next_sidepulse) {
        spawnEdgePulse(node, now);
        node.__sjake_next_sidepulse = now + 6 + Math.random() * 10; // next in 6..16s
    }

    // Draw existing pulses
    drawEdgePulses(ctx, node, x, y, w, h, theme, now);
}

function spawnEdgePulse(node, now) {
    const duration = 1.8 + Math.random() * 0.6; // seconds for edge->center
    const ease = 0.85; // ease exponent
    const amp = 1.0; // base alpha scale
    const width = 12; // band width in px
    node.__sjake_edge_pulses.push({ start: now, duration, ease, amp, width });
    // Cap pulses
    if (node.__sjake_edge_pulses.length > 3) node.__sjake_edge_pulses.shift();
}

function drawEdgePulses(ctx, node, x, y, w, h, theme, now) {
    const list = node.__sjake_edge_pulses || [];
    if (!list.length) return;
    const cx = x + w * 0.5;

    ctx.save();
    roundRect(ctx, x, y, w, h, 3);
    ctx.clip();

    for (let i = list.length - 1; i >= 0; i--) {
        const p = list[i];
        const t = (now - p.start) / p.duration;
        if (t >= 1) { list.splice(i, 1); continue; }
        const et = Math.pow(t, p.ease); // ease-in

        const bandW = p.width * (1 - 0.2 * et);
        const leftX = x + et * (w * 0.5 - bandW * 0.5);
        const rightX = x + w - et * (w * 0.5 - bandW * 0.5) - bandW;

        // Color uses title tint but stronger near start, fading toward center
        const alpha = (1 - t) * 0.38 * p.amp;
        const col = blendAlpha(theme.waveStrong, alpha / 0.35); // normalize vs waveStrong base

        // Left band
        let gL = ctx.createLinearGradient(leftX, 0, leftX + bandW, 0);
        gL.addColorStop(0.0, "rgba(0,0,0,0)");
        gL.addColorStop(0.5, col);
        gL.addColorStop(1.0, "rgba(0,0,0,0)");
        ctx.fillStyle = gL;
        ctx.fillRect(leftX, y - 2, bandW, h + 4);

        // Right band
        let gR = ctx.createLinearGradient(rightX, 0, rightX + bandW, 0);
        gR.addColorStop(0.0, "rgba(0,0,0,0)");
        gR.addColorStop(0.5, col);
        gR.addColorStop(1.0, "rgba(0,0,0,0)");
        ctx.fillStyle = gR;
        ctx.fillRect(rightX, y - 2, bandW, h + 4);
    }
    ctx.restore();
}
