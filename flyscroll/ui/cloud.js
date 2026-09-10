/* Static filament plate + phasic soma sparks (no per-edge glow). */
(function () {
  const LINE_VS = `
    attribute vec3 aPos;
    attribute vec3 aColor;
    uniform mat4 uMVP;
    varying vec3 vColor;
    varying float vDepth;
    void main() {
      vColor = aColor;
      vec4 clip = uMVP * vec4(aPos, 1.0);
      vDepth = clip.w;
      gl_Position = clip;
    }
  `;
  const LINE_FS = `
    precision mediump float;
    varying vec3 vColor;
    varying float vDepth;
    uniform float uBaseAlpha;
    void main() {
      float fog = clamp(1.25 / max(0.55, vDepth), 0.45, 1.0);
      gl_FragColor = vec4(vColor * fog, uBaseAlpha * fog);
    }
  `;
  const POINT_VS = `
    attribute vec3 aPos;
    attribute vec3 aColor;
    attribute float aAct;
    uniform mat4 uMVP;
    uniform float uPoint;
    varying vec3 vColor;
    varying float vAct;
    void main() {
      vAct = aAct;
      vColor = aColor;
      vec4 clip = uMVP * vec4(aPos, 1.0);
      gl_Position = clip;
      // Tight spark: readable over filaments, not a fat blob.
      float size = mix(1.4, 5.5, pow(clamp(aAct, 0.0, 1.0), 0.55));
      gl_PointSize = uPoint * size * (1.1 / max(0.35, clip.w));
    }
  `;
  const POINT_FS = `
    precision mediump float;
    varying vec3 vColor;
    varying float vAct;
    void main() {
      if (vAct < 0.18) discard;
      vec2 p = gl_PointCoord * 2.0 - 1.0;
      float r2 = dot(p, p);
      if (r2 > 1.0) discard;
      float core = exp(-r2 * 9.0);
      float rim = exp(-r2 * 3.5);
      float a = pow(clamp(vAct, 0.0, 1.0), 0.6);
      vec3 hot = vec3(1.0, 0.96, 0.88);
      vec3 col = mix(vColor, hot, 0.65 * a);
      float alpha = (0.9 * core + 0.35 * rim) * a;
      gl_FragColor = vec4(col, alpha);
    }
  `;

  function compile(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s) || "shader compile failed");
    }
    return s;
  }
  function program(gl, vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(p) || "link failed");
    }
    return p;
  }
  function locs(gl, prog, uniforms, attribs) {
    const u = {}, a = {};
    for (const n of uniforms) u[n] = gl.getUniformLocation(prog, n);
    for (const n of attribs) a[n] = gl.getAttribLocation(prog, n);
    return { u, a };
  }
  function mat4() { return new Float32Array(16); }
  function perspective(out, fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    out.fill(0);
    out[0] = f / aspect; out[5] = f;
    out[10] = (far + near) / (near - far); out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    return out;
  }
  function multiply(out, a, b) {
    const r = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let rI = 0; rI < 4; rI++) {
        r[c * 4 + rI] =
          a[rI] * b[c * 4] + a[4 + rI] * b[c * 4 + 1] +
          a[8 + rI] * b[c * 4 + 2] + a[12 + rI] * b[c * 4 + 3];
      }
    }
    out.set(r); return out;
  }
  function lookAt(out, eye, target, up) {
    const zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
    let len = Math.hypot(zx, zy, zz) || 1;
    const z0 = zx / len, z1 = zy / len, z2 = zz / len;
    let xx = up[1] * z2 - up[2] * z1;
    let xy = up[2] * z0 - up[0] * z2;
    let xz = up[0] * z1 - up[1] * z0;
    len = Math.hypot(xx, xy, xz) || 1;
    xx /= len; xy /= len; xz /= len;
    const y0 = z1 * xz - z2 * xy, y1 = z2 * xx - z0 * xz, y2 = z0 * xy - z1 * xx;
    out.set([
      xx, y0, z0, 0, xy, y1, z1, 0, xz, y2, z2, 0,
      -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
      -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]),
      -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]), 1,
    ]);
    return out;
  }
  function decodeB64(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  window.FlyCloud = function FlyCloud(canvas) {
    const gl = canvas.getContext("webgl", {
      antialias: false,
      alpha: false,
      premultipliedAlpha: false,
      powerPreference: "high-performance",
    });
    if (!gl) throw new Error("WebGL unavailable");

    const lineProg = program(gl, LINE_VS, LINE_FS);
    const pointProg = program(gl, POINT_VS, POINT_FS);
    const lineL = locs(gl, lineProg, ["uMVP", "uBaseAlpha"], ["aPos", "aColor"]);
    const pointL = locs(gl, pointProg, ["uMVP", "uPoint"], ["aPos", "aColor", "aAct"]);

    const linePosBuf = gl.createBuffer();
    const lineColBuf = gl.createBuffer();
    const pointPosBuf = gl.createBuffer();
    const pointColBuf = gl.createBuffer();
    const pointActBuf = gl.createBuffer();

    let nPoints = 0, nLineVerts = 0;
    let nodeAct = null;

    let yaw = 0.0, pitch = 0.18, radius = 1.55;
    let dragging = false, lastX = 0, lastY = 0;
    let lastXyz = null;
    const mvp = mat4(), proj = mat4(), view = mat4();
    let target = [0, 0, 0];
    let lastW = 0, lastH = 0;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.clearColor(0, 0, 0, 1);
    gl.disable(gl.DEPTH_TEST);

    function fitCamera(xyz) {
      if (!xyz || xyz.length < 3) return;
      lastXyz = xyz;
      const n = (xyz.length / 3) | 0;
      const xs = new Float32Array(n);
      const ys = new Float32Array(n);
      const zs = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        xs[i] = xyz[i * 3];
        ys[i] = xyz[i * 3 + 1];
        zs[i] = xyz[i * 3 + 2];
      }
      const pct = (arr, p) => {
        const a = Float32Array.from(arr).sort();
        const t = (a.length - 1) * p;
        const i0 = t | 0;
        const f = t - i0;
        return a[i0] * (1 - f) + a[Math.min(i0 + 1, a.length - 1)] * f;
      };
      const minX = pct(xs, 0.04), maxX = pct(xs, 0.96);
      const minY = pct(ys, 0.04), maxY = pct(ys, 0.96);
      const minZ = pct(zs, 0.08), maxZ = pct(zs, 0.92);
      target = [0.5 * (minX + maxX), 0.5 * (minY + maxY), 0.5 * (minZ + maxZ)];
      const dx = Math.max(maxX - minX, 0.15);
      const dy = Math.max(maxY - minY, 0.15);
      const dz = Math.max(maxZ - minZ, 0.15);
      const aspect = Math.max(0.5, (canvas.clientWidth || 1) / Math.max(canvas.clientHeight || 1, 1));
      const fovy = Math.PI / 3.0;
      const halfH = 0.5 * Math.max(dy, dz * 0.35);
      const halfW = 0.5 * dx;
      const needY = halfH / Math.tan(fovy * 0.5);
      const needX = halfW / (Math.tan(fovy * 0.5) * aspect);
      radius = Math.max(needX, needY) * 0.92;
      yaw = 0.0;
      pitch = 0.18;
    }

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      if (w === lastW && h === lastH) return;
      lastW = w; lastH = h;
      canvas.width = w;
      canvas.height = h;
      gl.viewport(0, 0, w, h);
    }

    function setAnatomy(anatomy) {
      const count = anatomy.points | 0;
      const xyzBytes = decodeB64(anatomy.xyz_b64);
      const xyz = new Float32Array(xyzBytes.buffer, xyzBytes.byteOffset, count * 3);
      nPoints = count;
      nodeAct = new Float32Array(count);
      fitCamera(xyz);

      let nodeRgb;
      if (anatomy.node_rgb_b64) {
        const nb = decodeB64(anatomy.node_rgb_b64);
        nodeRgb = new Uint8Array(nb.buffer, nb.byteOffset, count * 3);
      } else {
        nodeRgb = new Uint8Array(count * 3);
      }
      const nodeCol = new Float32Array(count * 3);
      for (let i = 0; i < count * 3; i++) nodeCol[i] = nodeRgb[i] / 255;

      gl.bindBuffer(gl.ARRAY_BUFFER, pointPosBuf);
      gl.bufferData(gl.ARRAY_BUFFER, xyz, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, pointColBuf);
      gl.bufferData(gl.ARRAY_BUFFER, nodeCol, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, pointActBuf);
      gl.bufferData(gl.ARRAY_BUFFER, nodeAct, gl.DYNAMIC_DRAW);

      nLineVerts = 0;
      if (anatomy.edges_b64 && anatomy.edge_rgb_b64 && anatomy.edges > 0) {
        const nEdges = anatomy.edges | 0;
        const eb = decodeB64(anatomy.edges_b64);
        const edgePairs = new Int32Array(eb.buffer, eb.byteOffset, nEdges * 2);
        const cb = decodeB64(anatomy.edge_rgb_b64);
        const edgeRgb = new Uint8Array(cb.buffer, cb.byteOffset, nEdges * 3);
        const linePos = new Float32Array(nEdges * 6);
        const lineCol = new Float32Array(nEdges * 6);

        for (let e = 0; e < nEdges; e++) {
          const a = edgePairs[e * 2], b = edgePairs[e * 2 + 1], o = e * 6;
          linePos[o] = xyz[a * 3]; linePos[o + 1] = xyz[a * 3 + 1]; linePos[o + 2] = xyz[a * 3 + 2];
          linePos[o + 3] = xyz[b * 3]; linePos[o + 4] = xyz[b * 3 + 1]; linePos[o + 5] = xyz[b * 3 + 2];
          const r = edgeRgb[e * 3] / 255, g = edgeRgb[e * 3 + 1] / 255, bl = edgeRgb[e * 3 + 2] / 255;
          lineCol[o] = r * 0.55; lineCol[o + 1] = g * 0.55; lineCol[o + 2] = bl * 0.55;
          lineCol[o + 3] = r * 0.55; lineCol[o + 4] = g * 0.55; lineCol[o + 5] = bl * 0.55;
        }
        nLineVerts = nEdges * 2;
        gl.bindBuffer(gl.ARRAY_BUFFER, linePosBuf);
        gl.bufferData(gl.ARRAY_BUFFER, linePos, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineColBuf);
        gl.bufferData(gl.ARRAY_BUFFER, lineCol, gl.STATIC_DRAW);
      }
    }

    function setActivity(actB64, count) {
      if (!count || count !== nPoints || !nodeAct) return;
      const bytes = decodeB64(actB64);
      const u8 = new Uint8Array(bytes.buffer, bytes.byteOffset, count);
      for (let i = 0; i < count; i++) nodeAct[i] = u8[i] * 0.00392156862745098;
      gl.bindBuffer(gl.ARRAY_BUFFER, pointActBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, nodeAct);
    }

    function draw() {
      resize();
      perspective(proj, Math.PI / 3.0, canvas.width / Math.max(canvas.height, 1), 0.05, 20);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const eye = [
        target[0] + radius * cp * sy,
        target[1] + radius * sp,
        target[2] + radius * cp * cy,
      ];
      lookAt(view, eye, target, [0, 1, 0]);
      multiply(mvp, proj, view);
      gl.clear(gl.COLOR_BUFFER_BIT);

      if (nLineVerts > 0) {
        gl.useProgram(lineProg);
        gl.uniformMatrix4fv(lineL.u.uMVP, false, mvp);
        gl.uniform1f(lineL.u.uBaseAlpha, 0.035);
        gl.bindBuffer(gl.ARRAY_BUFFER, linePosBuf);
        gl.enableVertexAttribArray(lineL.a.aPos);
        gl.vertexAttribPointer(lineL.a.aPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineColBuf);
        gl.enableVertexAttribArray(lineL.a.aColor);
        gl.vertexAttribPointer(lineL.a.aColor, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINES, 0, nLineVerts);
      }

      if (nPoints > 0) {
        gl.useProgram(pointProg);
        gl.uniformMatrix4fv(pointL.u.uMVP, false, mvp);
        gl.uniform1f(pointL.u.uPoint, Math.max(1.8, canvas.height / 280));
        gl.bindBuffer(gl.ARRAY_BUFFER, pointPosBuf);
        gl.enableVertexAttribArray(pointL.a.aPos);
        gl.vertexAttribPointer(pointL.a.aPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, pointColBuf);
        gl.enableVertexAttribArray(pointL.a.aColor);
        gl.vertexAttribPointer(pointL.a.aColor, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, pointActBuf);
        gl.enableVertexAttribArray(pointL.a.aAct);
        gl.vertexAttribPointer(pointL.a.aAct, 1, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, nPoints);
      }
    }

    canvas.addEventListener("pointerdown", (e) => {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      yaw += (e.clientX - lastX) * 0.005;
      pitch = Math.max(-0.15, Math.min(1.15, pitch + (e.clientY - lastY) * 0.005));
      lastX = e.clientX; lastY = e.clientY;
    });
    const stop = () => { dragging = false; };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener("dblclick", () => { if (lastXyz) fitCamera(lastXyz); });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      radius = Math.max(0.7, Math.min(5.0, radius * (e.deltaY > 0 ? 1.08 : 0.92)));
    }, { passive: false });

    (function loop() { draw(); requestAnimationFrame(loop); })();

    return {
      setAnatomy,
      setPositions(xyzB64, count) { setAnatomy({ points: count, xyz_b64: xyzB64, edges: 0 }); },
      setActivity,
    };
  };
})();
