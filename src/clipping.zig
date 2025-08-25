// Juste un test d'apprentissage de Zig, basé sur la partie graphique de mon émulateur Nintendo DS développé il y a 4 mois.

const std = @import("std");

const console_utf8 = @import("console_utf8.zig");

const WIDTH: usize = 256;
const HEIGHT: usize = 192;

const Mat4 = struct {
    // row-major
    p: [4][4]f32,

    fn identity() Mat4 {
        var m = std.mem.zeroes(Mat4);
        var i: usize = 0;
        while (i < 4) : (i += 1) m.p[i][i] = 1.0;
        return m;
    }

    fn mul(dst: *Mat4, src: Mat4) void {
        const a = dst.*;
        const b = src;
        var r = std.mem.zeroes(Mat4);
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            var j: usize = 0;
            while (j < 4) : (j += 1) {
                var k: usize = 0;
                var sum: f32 = 0;
                while (k < 4) : (k += 1) sum += a.p[i][k] * b.p[k][j];
                r.p[i][j] = sum;
            }
        }
        dst.* = r;
    }

    fn perspective(fov_deg: f32, aspect: f32, z_near: f32, z_far: f32) Mat4 {
        const pi: f32 = @as(f32, @floatCast(std.math.pi));
        const f: f32 = 1.0 / @tan((fov_deg * 0.5) * pi / 180.0);

        var m = std.mem.zeroes(Mat4);
        m.p[0][0] = f / aspect;
        m.p[1][1] = f;
        m.p[2][2] = (z_far + z_near) / (z_near - z_far);
        m.p[2][3] = (2.0 * z_far * z_near) / (z_near - z_far);
        m.p[3][2] = -1.0;
        return m;
    }

    fn translate(x: f32, y: f32, z: f32) Mat4 {
        var m = Mat4.identity();
        m.p[0][3] = x;
        m.p[1][3] = y;
        m.p[2][3] = z;
        return m;
    }

    fn rotateY(angle: f32) Mat4 {
        const c = @cos(angle);
        const s = @sin(angle);
        var m = Mat4.identity();
        m.p[0][0] = c;
        m.p[0][2] = s;
        m.p[2][0] = -s;
        m.p[2][2] = c;
        return m;
    }
};

const Vec4 = struct {
    p: [4]f32,

    fn mul(m: Mat4, v: Vec4) Vec4 {
        var r = Vec4{ .p = [_]f32{ 0, 0, 0, 0 } };
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            var k: usize = 0;
            var sum: f32 = 0;
            while (k < 4) : (k += 1) sum += m.p[i][k] * v.p[k];
            r.p[i] = sum;
        }
        return r;
    }
};

const Vertex = struct {
    v: Vec4, // homogène x,y,z,w
    r: f32, // 0..255 (pré-divisés par w après normalize)
    g: f32,
    b: f32,
};

const EdgeAttr = struct {
    x: i32,
    z: f32,
    w: f32,
    r: f32,
    g: f32,
    b: f32,
};

const Gpu = struct {
    rgb: [HEIGHT][WIDTH][3]u8,
    depth: [HEIGHT][WIDTH]f32,
    view_x: i32,
    view_y: i32,
    view_w: i32,
    view_h: i32,

    fn clear(self: *Gpu, R: u8, G: u8, B: u8, depth_val: f32) void {
        var y: usize = 0;
        while (y < HEIGHT) : (y += 1) {
            var x: usize = 0;
            while (x < WIDTH) : (x += 1) {
                self.rgb[y][x][0] = R;
                self.rgb[y][x][1] = G;
                self.rgb[y][x][2] = B;
                self.depth[y][x] = depth_val;
            }
        }
    }

    fn WritePpm(self: *const Gpu, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();

        var header_buf: [64]u8 = undefined;
        const header = try std.fmt.bufPrint(&header_buf, "P6\n{d} {d}\n255\n", .{ WIDTH, HEIGHT });
        try file.writeAll(header);
        try file.writeAll(std.mem.asBytes(&self.rgb));
    }
};

// Interpolation d'un sommet sur une coupe homogène
fn interpVertex(cur: Vertex, prev: Vertex, diffCur: f32, diffPrev: f32) Vertex {
    const ac = @abs(diffCur);
    const ap = @abs(diffPrev);
    const sum = ac + ap;

    var out = Vertex{
        .v = .{ .p = [_]f32{ 0, 0, 0, 0 } },
        .r = 0,
        .g = 0,
        .b = 0,
    };

    var k: usize = 0;
    while (k < 4) : (k += 1) {
        out.v.p[k] = (cur.v.p[k] * ap + prev.v.p[k] * ac) / sum;
    }
    out.r = (cur.r * ap + prev.r * ac) / sum;
    out.g = (cur.g * ap + prev.g * ac) / sum;
    out.b = (cur.b * ap + prev.b * ac) / sum;
    return out;
}

// Clip contre 7 plans: w±x>=0, w±y>=0, w±z>=0, et w>=0.
// vtxs: in/out, retourne 0 (dehors), -1 (pas clippé), >0 (nouveau N)
fn clipPolygon(vtxs: *[16]Vertex, n_in: i32) i32 {
    var n = n_in;
    var tmp: [16]Vertex = undefined;
    var clipped = false;

    var i: i32 = 6;
    while (i >= 0) : (i -= 1) {
        var nn: i32 = 0;
        var cur: i32 = 0;
        while (cur < n) : (cur += 1) {
            const prev: i32 = if (cur != 0) cur - 1 else n - 1;

            const ucur: usize = @as(usize, @intCast(cur));
            const uprev: usize = @as(usize, @intCast(prev));
            var dc = vtxs.*[ucur].v.p[3];
            var dp = vtxs.*[uprev].v.p[3];

            if ((i & 1) == 1) {
                const axis: usize = @as(usize, @intCast(@divTrunc(i, 2)));
                dc -= vtxs.*[ucur].v.p[axis];
                dp -= vtxs.*[uprev].v.p[axis];
            } else {
                const axis2: usize = @as(usize, @intCast(@divTrunc(i, 2)));
                dc += vtxs.*[ucur].v.p[axis2];
                dp += vtxs.*[uprev].v.p[axis2];
            }

            if (dc * dp < 0.0) {
                tmp[@as(usize, @intCast(nn))] =
                    interpVertex(vtxs.*[ucur], vtxs.*[uprev], dc, dp);
                nn += 1;
                clipped = true;
            }
            if (dc >= 0.0) {
                tmp[@as(usize, @intCast(nn))] = vtxs.*[ucur];
                nn += 1;
            }
        }

        n = nn;
        var j: i32 = 0;
        while (j < n) : (j += 1) {
            vtxs.*[@as(usize, @intCast(j))] = tmp[@as(usize, @intCast(j))];
        }
        if (n == 0) break;
    }

    if (!clipped) return if (n == 0) 0 else -1;
    return n;
}

fn normalizeVertices(v: *[16]Vertex, n: i32) void {
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        const idx: usize = @as(usize, @intCast(i));
        var w = v.*[idx].v.p[3];
        if (w == 0.0) w = 1e-6;

        v.*[idx].v.p[3] = 1.0 / w; // stocke 1/w
        v.*[idx].v.p[0] /= w;
        v.*[idx].v.p[1] /= w;
        v.*[idx].v.p[2] /= w;
        v.*[idx].r /= w;
        v.*[idx].g /= w;
        v.*[idx].b /= w;
    }
}

fn renderLineAttrs(
    gpu: *Gpu,
    v0: *const Vertex,
    v1: *const Vertex,
    left: *[HEIGHT]EdgeAttr,
    right: *[HEIGHT]EdgeAttr,
) void {
    const x0f = (v0.v.p[0] + 1.0) * @as(f32, @floatFromInt(gpu.view_w)) * 0.5 + @as(f32, @floatFromInt(gpu.view_x));
    const y0f = (1.0 - v0.v.p[1]) * @as(f32, @floatFromInt(gpu.view_h)) * 0.5 + @as(f32, @floatFromInt(gpu.view_y));
    const x1f = (v1.v.p[0] + 1.0) * @as(f32, @floatFromInt(gpu.view_w)) * 0.5 + @as(f32, @floatFromInt(gpu.view_x));
    const y1f = (1.0 - v1.v.p[1]) * @as(f32, @floatFromInt(gpu.view_h)) * 0.5 + @as(f32, @floatFromInt(gpu.view_y));

    var x0 = @as(i32, @intFromFloat(x0f));
    var y0 = @as(i32, @intFromFloat(y0f));
    var x1 = @as(i32, @intFromFloat(x1f));
    var y1 = @as(i32, @intFromFloat(y1f));

    var a0 = EdgeAttr{ .x = 0, .z = v0.v.p[2], .w = v0.v.p[3], .r = v0.r, .g = v0.g, .b = v0.b };
    var a1 = EdgeAttr{ .x = 0, .z = v1.v.p[2], .w = v1.v.p[3], .r = v1.r, .g = v1.g, .b = v1.b };

    const dx = x1 - x0;
    const m = (y1f - y0f) / (@as(f32, @floatFromInt(if (dx == 0) 1 else dx)));

    if (@abs(m) > 1.0) {
        const invm = 1.0 / m;
        if (y0 > y1) {
            std.mem.swap(i32, &y0, &y1);
            std.mem.swap(i32, &x0, &x1);
            std.mem.swap(EdgeAttr, &a0, &a1);
        }

        const h = y1 - y0;
        if (h == 0) return;

        var x = @as(f32, @floatFromInt(x0));
        const di = EdgeAttr{
            .x = 0,
            .z = (a1.z - a0.z) / @as(f32, @floatFromInt(h)),
            .w = (a1.w - a0.w) / @as(f32, @floatFromInt(h)),
            .r = (a1.r - a0.r) / @as(f32, @floatFromInt(h)),
            .g = (a1.g - a0.g) / @as(f32, @floatFromInt(h)),
            .b = (a1.b - a0.b) / @as(f32, @floatFromInt(h)),
        };

        var y: i32 = y0;
        while (y <= y1) : (y += 1) {
            if (y >= 0 and y < @as(i32, @intCast(HEIGHT))) {
                var sx = @as(i32, @intFromFloat(x));
                if (sx < 0) sx = 0;
                if (sx >= @as(i32, @intCast(WIDTH))) sx = @as(i32, @intCast(WIDTH - 1));

                var cur = a0;
                cur.x = sx;

                const uy = @as(usize, @intCast(y));
                if (sx <= left.*[uy].x) left.*[uy] = cur;
                if (sx >= right.*[uy].x) right.*[uy] = cur;
            }

            x += invm;
            a0.z += di.z;
            a0.w += di.w;
            a0.r += di.r;
            a0.g += di.g;
            a0.b += di.b;
        }
    } else {
        if (x0 > x1) {
            std.mem.swap(i32, &x0, &x1);
            std.mem.swap(i32, &y0, &y1);
            std.mem.swap(EdgeAttr, &a0, &a1);
        }
        const w = x1 - x0;
        if (w == 0) return;

        var y = @as(f32, @floatFromInt(y0));
        const di = EdgeAttr{
            .x = 0,
            .z = (a1.z - a0.z) / @as(f32, @floatFromInt(w)),
            .w = (a1.w - a0.w) / @as(f32, @floatFromInt(w)),
            .r = (a1.r - a0.r) / @as(f32, @floatFromInt(w)),
            .g = (a1.g - a0.g) / @as(f32, @floatFromInt(w)),
            .b = (a1.b - a0.b) / @as(f32, @floatFromInt(w)),
        };

        var x: i32 = x0;
        while (x <= x1) : (x += 1) {
            const sx = x;
            const sy = @as(i32, @intFromFloat(y));

            if (sy >= 0 and sy < @as(i32, @intCast(HEIGHT))) {
                var clamped_sx = sx;
                if (clamped_sx < 0) clamped_sx = 0;
                if (clamped_sx >= @as(i32, @intCast(WIDTH))) clamped_sx = @as(i32, @intCast(WIDTH - 1));

                var cur = a0;
                cur.x = clamped_sx;

                const uy = @as(usize, @intCast(sy));
                if (clamped_sx <= left.*[uy].x) left.*[uy] = cur;
                if (clamped_sx >= right.*[uy].x) right.*[uy] = cur;
            }

            y += m;
            a0.z += di.z;
            a0.w += di.w;
            a0.r += di.r;
            a0.g += di.g;
            a0.b += di.b;
        }
    }
}

fn drawPolygon(gpu: *Gpu, poly: []const Vertex) void {
    // bornes verticales
    var y_min: i32 = @as(i32, @intCast(HEIGHT));
    var y_max: i32 = -1;

    var i: usize = 0;
    while (i < poly.len) : (i += 1) {
        const y = @as(i32, @intFromFloat((1.0 - poly[i].v.p[1]) * @as(f32, @floatFromInt(gpu.view_h)) * 0.5 + @as(f32, @floatFromInt(gpu.view_y))));
        if (y < y_min) y_min = y;
        if (y > y_max) y_max = y;
    }
    if (y_min < 0) y_min = 0;
    y_max = @min(y_max, @as(i32, @intCast(HEIGHT - 1)));
    if (y_max < y_min) return;

    var left: [HEIGHT]EdgeAttr = undefined;
    var right: [HEIGHT]EdgeAttr = undefined;

    var y: i32 = y_min;
    while (y <= y_max) : (y += 1) {
        left[@as(usize, @intCast(y))].x = @as(i32, @intCast(WIDTH));
        right[@as(usize, @intCast(y))].x = -1;
    }

    var e: usize = 0;
    while (e < poly.len) : (e += 1) {
        const j = if (e + 1 == poly.len) 0 else e + 1;
        renderLineAttrs(gpu, &poly[e], &poly[j], &left, &right);
    }

    y = y_min;
    while (y <= y_max) : (y += 1) {
        const uy = @as(usize, @intCast(y));
        if (right[uy].x < left[uy].x) continue;

        const L = left[uy].x;
        const R = right[uy].x;
        const span_i = R - L + 1;
        if (span_i <= 0) continue;

        var a = left[uy];
        const span = @as(f32, @floatFromInt(span_i));

        const di = EdgeAttr{
            .x = 0,
            .z = (right[uy].z - a.z) / span,
            .w = (right[uy].w - a.w) / span,
            .r = (right[uy].r - a.r) / span,
            .g = (right[uy].g - a.g) / span,
            .b = (right[uy].b - a.b) / span,
        };

        var x: i32 = L;
        while (x <= R) : (x += 1) {
            const ux = @as(usize, @intCast(std.math.clamp(x, 0, @as(i32, @intCast(WIDTH - 1)))));
            // z déjà divisé par w après normalize ; w = 1/w
            const z = a.z;
            const iw = a.w;

            if (z < gpu.depth[uy][ux]) {
                var r = a.r / iw;
                var g = a.g / iw;
                var b = a.b / iw;
                r = std.math.clamp(r, 0.0, 255.0);
                g = std.math.clamp(g, 0.0, 255.0);
                b = std.math.clamp(b, 0.0, 255.0);

                gpu.rgb[uy][ux][0] = @as(u8, @intFromFloat(r));
                gpu.rgb[uy][ux][1] = @as(u8, @intFromFloat(g));
                gpu.rgb[uy][ux][2] = @as(u8, @intFromFloat(b));
                gpu.depth[uy][ux] = z;
            }

            a.z += di.z;
            a.w += di.w;
            a.r += di.r;
            a.g += di.g;
            a.b += di.b;
        }
    }
}

pub fn main() !void {
    console_utf8.setUtf8Console();

    var gpu = Gpu{
        .rgb = undefined,
        .depth = undefined,
        .view_x = 0,
        .view_y = 0,
        .view_w = @as(i32, @intCast(WIDTH)),
        .view_h = @as(i32, @intCast(HEIGHT)),
    };
    gpu.clear(20, 20, 28, 1e9);

    // Matrices
    const proj = Mat4.perspective(
        70.0,
        @as(f32, @floatFromInt(WIDTH)) / @as(f32, @floatFromInt(HEIGHT)),
        0.1,
        100.0,
    );
    var model = Mat4.identity();
    const rot = Mat4.rotateY(0.7);
    const trn = Mat4.translate(0.0, 0.0, -3.0);
    model = rot;
    Mat4.mul(&model, trn); // model = rot * translate

    var clip = proj;
    Mat4.mul(&clip, model); // clip = proj * model

    // Triangle RGB
    var local: [3]Vertex = .{
        .{ .v = .{ .p = .{ -0.8, -0.8, 0.0, 1.0 } }, .r = 255, .g = 60, .b = 60 },
        .{ .v = .{ .p = .{ 0.8, -0.8, 0.0, 1.0 } }, .r = 60, .g = 255, .b = 60 },
        .{ .v = .{ .p = .{ 0.0, 0.8, 0.0, 1.0 } }, .r = 60, .g = 60, .b = 255 },
    };

    // Applique clip (proj*model)
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        const v = Vec4{ .p = .{ local[i].v.p[0], local[i].v.p[1], local[i].v.p[2], 1.0 } };
        local[i].v = Vec4.mul(clip, v);
    }

    // Clipping (peut créer de nouveaux sommets)
    var work: [16]Vertex = undefined;
    work[0] = local[0];
    work[1] = local[1];
    work[2] = local[2];

    var n = clipPolygon(&work, 3);
    if (n == 0) {
        std.debug.print("Triangle entièrement hors du volume de vue.\n", .{});
        gpu.clear(0, 0, 0, 1e9);
    } else {
        if (n == -1) {
            n = 3;
            work[0] = local[0];
            work[1] = local[1];
            work[2] = local[2];
        }

        normalizeVertices(&work, n);

        // Construit une slice de sommets
        var poly_store: [8]Vertex = undefined;
        const count: usize = @as(usize, @intCast(n));
        i = 0;
        while (i < count) : (i += 1) poly_store[i] = work[i];
        var poly = poly_store[0..count];

        // Culling optionnel (conserve CCW)
        const ax = poly[1].v.p[0] - poly[0].v.p[0];
        const ay = poly[1].v.p[1] - poly[0].v.p[1];
        const bx = poly[count - 1].v.p[0] - poly[0].v.p[0];
        const by = poly[count - 1].v.p[1] - poly[0].v.p[1];
        const area = ax * by - ay * bx;

        if (area <= 0.0) {
            // inverse l’ordre in-place
            var a: usize = 0;
            while (a < count / 2) : (a += 1) {
                std.mem.swap(Vertex, &poly_store[a], &poly_store[count - 1 - a]);
            }
            poly = poly_store[0..count];
        }

        drawPolygon(&gpu, poly);
    }

    try gpu.WritePpm("out.ppm");
    std.debug.print("Image écrite dans out.ppm ({d}x{d})\n", .{ WIDTH, HEIGHT });
}
